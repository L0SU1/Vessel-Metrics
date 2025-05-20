"""
vessel_analysis_modular.py

A command-line tool to analyze 3D vessel masks by skeletonizing,
building a graph, extracting segments, and computing only requested metrics.

USAGE:
    python vessel_analysis_modular.py <input_path> [--min_size INT] [--metrics METRIC [METRIC ...]] [--output_folder PATH]

ARGUMENTS:
    input_path       Path to the NIfTI vessel mask (.nii or .nii.gz) or to the .npy.
    --min_size       (optional) Minimum size (in voxels) to keep connected components
                     after cleaning small objects. Default is 32.
    --metrics        (optional) List of metrics to compute/display. Options include:
                     total_length, bifurcation_density, volume, fractal_dimension,
                     lacunarity, geodesic_length, avg_diameter,
                     spline_arc_length, spline_chord_length, mean_curvature,
                     mean_square_curvature, rms_curvature, arc_over_chord, rmse.
                     Default is all metrics.
    --output_folder  (optional) Path to save results. Default is './VESSEL METRICS'.
"""
import os
import glob
import argparse
import csv
import nibabel as nib
import numpy as np
import pandas as pd
import networkx as nx
from skimage.morphology import skeletonize, remove_small_objects
from scipy.ndimage import distance_transform_edt, label as ndi_label
from scipy.interpolate import splprep, splev, interp1d
from sklearn.linear_model import LinearRegression
from collections import defaultdict





def compute_tortuosity_metrics(points, smoothing=0, n_samples=500, counts=None):
    """
    Compute tortuosity metrics for a 3D curve defined by 'points' using a cubic B-spline.
    This function reparameterizes the spline by arc length to ensure a uniform-speed curve,
    and can optionally down-weight curvature by per-point occurrence counts.

    Parameters:
      points : array-like, shape (N, 3)
        Input list of 3D curve points.
      smoothing : float, optional
        Smoothing factor for spline fitting (default: 0).
      n_samples : int, optional
        Number of samples along the curve for evaluation (default: 500).
      counts : array-like, shape (N,), optional
        Occurrence counts n(x) at each original input point.  After
        interpolation this yields n(s) at each sampled s, and we will
        weight curvature as κ(s)/n(s).

    Returns:
      dict of tortuosity metrics:
          - spline_arc_length
          - spline_chord_length
          - spline_mean_curvature       (weighted)
          - spline_mean_square_curvature (weighted)
          - spline_rms_curvature        (weighted)
          - arc_over_chord
          - fit_rmse
    """
    pts = np.asarray(points)
    if pts.shape[0] < 4:
        # Not enough points to fit a cubic B-spline
        nan_dict = {k: np.nan for k in [
            'spline_arc_length','spline_chord_length',
            'spline_mean_curvature','spline_mean_square_curvature',
            'spline_rms_curvature','arc_over_chord','fit_rmse']}
        return nan_dict

    # 1) Fit spline and get original u-parameters
    tck, u = splprep(pts.T, s=smoothing)

    # 2) Dense evaluation to compute arc length
    u_fine   = np.linspace(0, 1, n_samples)
    deriv1   = np.array(splev(u_fine, tck, der=1)).T
    du       = np.gradient(u_fine)
    ds       = np.linalg.norm(deriv1, axis=1) * du
    s_cum    = np.cumsum(ds) - ds[0]
    arc_len  = s_cum[-1]

    # 3) Reparameterize by arc length → uniform s samples
    u_of_s   = interp1d(s_cum, u_fine, kind='linear',
                        bounds_error=False, fill_value=(0,1))
    s_uniform= np.linspace(0, arc_len, n_samples)
    u_uniform= u_of_s(s_uniform)
    pts_u    = np.array(splev(u_uniform, tck)).T

    # 4) Compute derivatives wrt s
    dt  = s_uniform[1] - s_uniform[0]
    d1  = np.gradient(pts_u, dt, axis=0)
    d2  = np.gradient(d1, dt, axis=0)

    # 5) Build the weight function n(s) by interpolating original counts if given
    if counts is not None:
        counts_orig  = np.asarray(counts)
        interp_cnt   = interp1d(u, counts_orig, kind='linear',
                                bounds_error=False,
                                fill_value=(counts_orig[0], counts_orig[-1]))
        n_s          = interp_cnt(u_uniform)
        n_s          = np.where(n_s <= 0, 1, n_s)  # clamp to ≥1
    else:
        n_s = np.ones(n_samples)

    # 6) Compute the standard curvature κ(s)
    cross_vec = np.cross(d1, d2)
    speed     = np.linalg.norm(d1, axis=1)
    # add epsilon to avoid div‐by‐zero in speed**3
    curvature = np.linalg.norm(cross_vec, axis=1) / (speed**3 + 1e-10)

    # 7) Form the weighted curvature and its square
    curv_w    = curvature / n_s
    curv2_w   = (curvature**2) / (n_s**2)

    # 8) Integrate weighted curvature over s
    mean_curv       = np.trapz(curv_w,  s_uniform)
    mean_sq_curv    = np.trapz(curv2_w, s_uniform)
    rms_curv        = np.sqrt(mean_sq_curv / arc_len) if arc_len>0 else 0

    # 9) Chord length & fit RMSE
    chord_len = np.linalg.norm(pts_u[-1] - pts_u[0])
    spline_at_u = np.array(splev(u, tck)).T
    fit_rmse    = np.sqrt(np.mean(np.sum((spline_at_u - pts)**2, axis=1)))

    return {
        'spline_arc_length':           arc_len,
        'spline_chord_length':         chord_len,
        'spline_mean_curvature':       mean_curv,
        'spline_mean_square_curvature':mean_sq_curv,
        'spline_rms_curvature':        rms_curv,
        'arc_over_chord':              (arc_len / chord_len) if chord_len>0 else np.inf,
        'fit_rmse':                    fit_rmse
    }



def fractal_dimension(points, box_sizes=None):
    """
    Compute fractal dimension using the box counting method for a set of 3D points.
    'points' is an array of shape (N, 3). Returns the fractal dimension.
    """
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        return np.nan

    # Shift points to positive coordinates
    mins = points.min(axis=0)
    shifted = points - mins
    maxs = shifted.max(axis=0)
    max_dim = max(maxs)

    # Guard against zero‐extent point clouds:
    if max_dim == 0:
        return np.nan

    # Define box sizes logarithmically if not provided
    if box_sizes is None:
        # Use 10 sizes from a fraction of max_dim to max_dim
        box_sizes = np.logspace(
            np.log10(max_dim / 50.0), 
            np.log10(max_dim), 
            num=10,
            base=10.0
        )
    
    counts = []
    for size in box_sizes:
        if size <= 0 or np.isnan(size):
            # Skip invalid sizes
            continue
        # Determine the number of boxes in each dimension
        bins = np.ceil(maxs / size).astype(int) + 1
        # Compute box indices
        indices = np.floor(shifted / size).astype(int)
        # Unique boxes that contain at least one point
        unique_boxes = {tuple(idx) for idx in indices}
        counts.append(len(unique_boxes))
    
    # If we couldn't collect any valid counts, bail out
    if len(counts) == 0:
        return np.nan

    # Fit a line to the log-log plot of (1/box_size) vs counts
    X = np.log(1.0 / np.array(box_sizes[:len(counts)])).reshape(-1, 1)
    y = np.log(counts)
    reg = LinearRegression().fit(X, y)
    return reg.coef_[0]



def calculate_lacunarity(points, box_size):
    """
    Estimate lacunarity for a set of 3D points using a grid of a given box_size.
    Here we build a grid covering the points and calculate the mean and variance
    of the count of points per box.
    """
    points = np.asarray(points)
    mins = points.min(axis=0)
    shifted = points - mins
    maxs = shifted.max(axis=0)
    # Determine number of boxes per dimension
    num_boxes = np.ceil(maxs / box_size).astype(int) + 1
    grid = np.zeros(num_boxes)
    
    # For each point, increment its corresponding box
    indices = np.floor(shifted / box_size).astype(int)
    for idx in indices:
        grid[tuple(idx)] += 1
    counts = grid.flatten()
    mean_val = counts.mean()
    var_val = counts.var()
    # A common lacunarity measure:
    lac = var_val / (mean_val**2) + 1 if mean_val != 0 else np.nan
    return lac

def analyze_component_structure(G_comp):
    """
    Calculates the number of loops and the number of nodes with abnormal degree (> 3)
    in a connected component.

    Parameters:
        G_comp: networkx.Graph
            A subgraph representing a connected component.

    Returns:
        tuple: (num_loops, num_abnormal_degree_nodes)
    """
    num_loops = len(nx.cycle_basis(G_comp))
    abnormal_degree_nodes = [node for node, degree in G_comp.degree() if degree > 3]
    num_abnormal_degree_nodes = len(abnormal_degree_nodes)
    return num_loops, num_abnormal_degree_nodes






def extract_segments_from_component_using_shortest_path(G_comp, distance_map):
    """
    From the connected component G_comp, identify all endpoints (degree == 1)
    and select three roots:
      1. Endpoint with the largest diameter
      2. Endpoint with the second-largest diameter
      3. Bifurcation node (degree >= 3) with the largest diameter

    Then, compute shortest-path segments from each root to all other endpoints,
    and for each root, return both:
      - A list of shortest-path segments (each a list of nodes) from the root
        to every other endpoint.
      - A mapping from each node to the number of times it appears across those segments.

    Parameters:
      G_comp : networkx.Graph
          A subgraph representing a connected component.
      distance_map : dict-like
          Mapping from node (as tuple) to its radius value (in same units as graph coords).

    Returns:
      segment_lists : List[List[List[node]]]
          A list of three lists, each containing shortest-path segments
          (node lists) from one selected root to every other endpoint.
      segment_counts : List[Dict[node, int]]
          A list of three dictionaries, each corresponding to one selected root.
          Each dictionary maps nodes to their frequency of occurrence in all
          shortest-path segments from that root to every other endpoint.
    """
    # Identify endpoints (degree == 1)
    endpoints = [node for node, deg in G_comp.degree() if deg == 1]
    if not endpoints:
        return [[], [], []], [{}, {}, {}]

    # Diameter at a node = 2 * radius
    diameters_eps = {node: 2 * distance_map[tuple(node)] for node in endpoints}
    # Sort endpoints by diameter descending
    sorted_eps = sorted(endpoints, key=lambda n: diameters_eps[n], reverse=True)

    # First two roots: largest and second-largest diameter endpoints
    root1 = sorted_eps[0]
    root2 = sorted_eps[1] if len(sorted_eps) > 1 else root1

    # Third root: bifurcation (degree >= 3) with largest diameter, or fallback
    bif_nodes = [node for node, deg in G_comp.degree() if deg >= 3]
    if bif_nodes:
        diameters_bif = {node: 2 * distance_map[tuple(node)] for node in bif_nodes}
        root3 = max(bif_nodes, key=lambda n: diameters_bif[n])
    else:
        root3 = root1

    roots = [root1, root2, root3]
    segment_lists = []
    segment_counts_list = []

    # For each selected root, compute segments and counts
    for root in roots:
        segs = []
        counts = defaultdict(int)
        for ep in endpoints:
            if ep == root:
                continue
            try:
                path = nx.shortest_path(G_comp, source=root, target=ep)
            except nx.NetworkXNoPath:
                continue
            segs.append(path)
            # Count occurrences of each node along the path
            for node in path:
                counts[node] += 1
        segment_lists.append(segs)
        segment_counts_list.append(dict(counts))

    return segment_lists, segment_counts_list



def build_graph(skeleton):
    G = nx.Graph()
    shape = skeleton.shape
    fibers = np.argwhere(skeleton)
    for v in fibers:
        coord = tuple(v)
        G.add_node(coord)
        x, y, z = coord
        for i in range(max(0, x-1), min(shape[0], x+2)):
            for j in range(max(0, y-1), min(shape[1], y+2)):
                for k in range(max(0, z-1), min(shape[2], z+2)):
                    if (i, j, k) != (x, y, z) and skeleton[i, j, k]:
                        G.add_edge(coord, (i, j, k), weight=np.linalg.norm(np.array(coord) - np.array((i, j, k))))
    return G



def prune_graph(G):
    """
    Prune G by detecting all simple cycles of length 3 (triangles) and removing the
    heaviest edge in each triangle (based on 'weight').
    Modifies G in-place and returns it.
    """
    # Find all simple cycles in the graph
    loops = nx.cycle_basis(G)
    for cycle in loops:
        if len(cycle) != 3:
            continue

        # Identify the heaviest edge in the triangle
        max_edge = None
        max_weight = float('-inf')
        for i in range(3):
            u = cycle[i]
            v = cycle[(i + 1) % 3]

            # safely get weight from either direction
            if G.has_edge(u, v):
                w = G[u][v].get('weight', 0)
            elif G.has_edge(v, u):
                w = G[v][u].get('weight', 0)
            else:
                # no such edge—skip
                continue

            if w > max_weight:
                max_weight = w
                max_edge = (u, v)

        # Remove the heaviest edge if it still exists
        if max_edge:
            u, v = max_edge
            if G.has_edge(u, v):
                G.remove_edge(u, v)
            elif G.has_edge(v, u):
                G.remove_edge(v, u)

    return G


def aggregate_segmentation_metrics(output_folder):
    """
    Scan all Conn_comp_* subfolders under output_folder, read each
    General_metrics.csv, and produce a single segmentation_metrics.csv
    with:
      - total_length
      - num_bifurcations
      - volume
      - num_loops
      - num_abnormal_degree_nodes
      - root1_mean_curvature
      - root1_mean_square_curvature
      - root2_mean_curvature
      - root2_mean_square_curvature
      - root3_mean_curvature
      - root3_mean_square_curvature
      - grand_total_length
      - num_components

    For each tortuosity field, components missing that value are simply
    skipped (both from numerator and denominator).
    """
    comp_dirs = sorted(glob.glob(os.path.join(output_folder, 'Conn_comp_*')))
    num_components = len(comp_dirs)

    # fields to sum directly
    general_fields = [
        'total_length', 'num_bifurcations', 'volume',
        'num_loops', 'num_abnormal_degree_nodes'
    ]
    # tortuosity fields to weight
    tort_fields = [
        'root1_mean_curvature','root1_mean_square_curvature',
        'root2_mean_curvature','root2_mean_square_curvature',
        'root3_mean_curvature','root3_mean_square_curvature'
    ]

    # accumulators
    accum = {f: 0.0 for f in general_fields}
    weighted_sums = {f: 0.0 for f in tort_fields}
    length_sums   = {f: 0.0 for f in tort_fields}
    grand_total_length = 0.0

    for comp_dir in comp_dirs:
        csv_path = os.path.join(comp_dir, 'General_metrics.csv')
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)
        if df.empty:
            continue

        # fetch this component's length (0 if missing/NaN)
        tl = df.at[0, 'total_length'] if 'total_length' in df.columns else 0.0
        if pd.isna(tl):
            tl = 0.0

        grand_total_length += tl

        # general fields
        for f in general_fields:
            val = df.at[0, f] if f in df.columns and not pd.isna(df.at[0, f]) else 0.0
            accum[f] += val

        # tortuosity fields: only if present and not NaN/blank
        for f in tort_fields:
            if f in df.columns:
                val = df.at[0, f]
                if not pd.isna(val):
                    weighted_sums[f] += val * tl
                    length_sums[f]   += tl

    # finalize: merge general accum + tortuosity + summaries
    result = {f: accum[f] for f in general_fields}
    for f in tort_fields:
        if length_sums[f] > 0:
            result[f] = weighted_sums[f] / length_sums[f]
        else:
            result[f] = float('nan')

    result['grand_total_length'] = grand_total_length
    result['num_components']     = num_components

    out_df = pd.DataFrame([result])
    out_csv = os.path.join(output_folder, 'Whole_mask_metrics.csv')
    out_df.to_csv(out_csv, index=False)



def save_results(results, output_folder, save_masks=True):
    os.makedirs(output_folder, exist_ok=True)

    # Keys to include from the top‐level general data
    general_keys = [
        'total_length', 'num_bifurcations', 'bifurcation_density', 'volume',
        'fractal_dimension', 'lacunarity',
        'num_loops', 'num_abnormal_degree_nodes'
    ]

    # Sort components by total_length descending
    sorted_items = sorted(
        results.items(),
        key=lambda item: item[1].get('total_length', 0),
        reverse=True
    )

    for new_idx, (cid, data) in enumerate(sorted_items):
        comp_idx = new_idx + 1
        comp_dir = os.path.join(output_folder, f"Conn_comp_{comp_idx}")
        os.makedirs(comp_dir, exist_ok=True)

        # — General metrics + aggregated tortuosity —
        gen_data = {k: data.get(k, np.nan) for k in general_keys}

        # Pull in aggregated tortuosity if present
        # Expected format: data['aggregated_tortuosity_by_root'] is a list of dicts
        # with keys 'mean_curvature' and 'mean_square_curvature'
        agg = data.get('aggregated_tortuosity_by_root', [])
        for i in range(3):
            prefix = f'root{i+1}_'
            if i < len(agg):
                gen_data[prefix + 'mean_curvature']        = agg[i].get('mean_curvature', np.nan)
                gen_data[prefix + 'mean_square_curvature'] = agg[i].get('mean_square_curvature', np.nan)
            else:
                gen_data[prefix + 'mean_curvature']        = np.nan
                gen_data[prefix + 'mean_square_curvature'] = np.nan

        # Write the combined General_metrics.csv
        if gen_data:
            pd.DataFrame([gen_data]).to_csv(
                os.path.join(comp_dir, 'General_metrics.csv'),
                index=False
            )

        # — Component skeleton —
        if 'reconstructed_conn_comp' in data:
            nib.save(
                data['reconstructed_conn_comp'],
                os.path.join(comp_dir, f'Conn_comp_{comp_idx}_skeleton.nii.gz')
            )

        # — Segments as before —
        if 'segments_by_root' in data:
            segs_dir = os.path.join(comp_dir, 'Segments')
            os.makedirs(segs_dir, exist_ok=True)
            root_names = [
                'Largest endpoint root',
                'Second largest endpoint root',
                'Largest bifurcation root'
            ]
            for root_idx, root_entry in enumerate(data['segments_by_root']):
                root_dir = os.path.join(segs_dir, root_names[root_idx])
                os.makedirs(root_dir, exist_ok=True)

                metrics_list = root_entry.get('segment_metrics', [])
                masks_list   = root_entry.get('segment_masks', []) if save_masks else []

                for seg_idx, sm in enumerate(metrics_list, start=1):
                    seg_dir = os.path.join(root_dir, f"Segment_{seg_idx}")
                    os.makedirs(seg_dir, exist_ok=True)

                    # per-segment metrics
                    pd.DataFrame([sm]).to_csv(
                        os.path.join(seg_dir, 'Segment_metrics.csv'), index=False
                    )

                    # per-segment mask
                    if save_masks and seg_idx-1 < len(masks_list):
                        nib.save(
                            masks_list[seg_idx-1],
                            os.path.join(seg_dir, 'Segment.nii.gz')
                        )




def process(mask_path, min_size, selected_metrics, save_masks=True):
    """
    Build a `results` dict for each connected component, filling in:
      - top‐level general metrics (total_length, num_bifurcations, etc.)
      - 'reconstructed_conn_comp': the binary skeleton mask
      - 'segments_by_root': per‐segment metrics & masks
      - 'aggregated_tortuosity_by_root': weighted mean curvature & mean_square_curvature
    Does NOT write any files—use save_results() to export.
    """
    # 1) Load & threshold
    ext = os.path.splitext(mask_path)[1].lower()
    if ext in ('.nii', '.gz'):
        img = nib.load(mask_path)
        arr = img.get_fdata() > 0
        affine, header = img.affine, img.header
    else:
        loaded = np.load(mask_path)
        arr = (next(iter(loaded.values())) if isinstance(loaded, dict) else loaded) > 0
        affine, header = np.eye(4), nib.Nifti1Header()

    # 2) Clean, skeletonize, dist map, graph
    clean    = remove_small_objects(arr, min_size=min_size)
    skel     = skeletonize(clean)
    dist_map = distance_transform_edt(clean)
    G        = prune_graph(build_graph(skel))

    # Flags for which metric groups to compute
    need_general  = any(m in selected_metrics for m in [
        'total_length','num_bifurcations','bifurcation_density','volume'])
    need_fractal  = 'fractal_dimension' in selected_metrics
    need_lac      = 'lacunarity' in selected_metrics
    need_struct   = any(m in selected_metrics for m in [
        'num_loops','num_abnormal_degree_nodes'])
    need_segments = any(m in selected_metrics for m in [
        'geodesic_length','avg_diameter',
        'spline_mean_curvature','spline_mean_square_curvature'
    ])

    results = {}

    # 3) Process each connected component
    for cid, comp_nodes in enumerate(nx.connected_components(G)):
        Gc   = G.subgraph(comp_nodes)
        data = {}

        # Save the reconstructed component skeleton if desired
        
        
        vessel_mask = np.zeros_like(skel, dtype=bool)
        for node in Gc.nodes:
            r = dist_map[node]
            if r > 0:
                rr = int(np.ceil(r))
                x0, y0, z0 = node
                for dx in range(-rr, rr + 1):
                    for dy in range(-rr, rr + 1):
                        for dz in range(-rr, rr + 1):
                            x, y, z = x0 + dx, y0 + dy, z0 + dz
                            if (0 <= x < vessel_mask.shape[0] and
                                0 <= y < vessel_mask.shape[1] and
                                0 <= z < vessel_mask.shape[2]):
                                if np.sqrt(dx**2 + dy**2 + dz**2) <= r:
                                    vessel_mask[x, y, z] = True
        data['reconstructed_conn_comp'] = nib.Nifti1Image(vessel_mask.astype(np.uint8), affine, header)

        # — General metrics —
        if need_general:
            num_bif   = sum(1 for _,deg in Gc.degree() if deg>=3)
            total_len = sum(d['weight'] for *_,d in Gc.edges(data=True))
            if 'num_bifurcations'    in selected_metrics:
                data['num_bifurcations'] = num_bif
            if 'total_length'        in selected_metrics:
                data['total_length'] = total_len
            if 'bifurcation_density' in selected_metrics:
                data['bifurcation_density'] = num_bif/total_len if total_len>0 else np.nan
            if 'volume' in selected_metrics:
                vol = 0.0
                for u,v,d in Gc.edges(data=True):
                    r_avg = (dist_map[u]+dist_map[v]) / 2
                    vol += np.pi*(r_avg**2)*d['weight']
                data['volume'] = vol

        # — Fractal & Lacunarity —
        coords = np.array(list(Gc.nodes()))
        if need_fractal and 'fractal_dimension' in selected_metrics:
            data['fractal_dimension'] = fractal_dimension(coords)
        if need_lac and 'lacunarity' in selected_metrics:
            box_size = np.max(coords.max(axis=0)-coords.min(axis=0))/10 or 1
            data['lacunarity'] = calculate_lacunarity(coords, box_size)

        # — Structural —
        if need_struct:
            nl, nab = analyze_component_structure(Gc)
            if 'num_loops'                in selected_metrics:
                data['num_loops']               = nl
            if 'num_abnormal_degree_nodes' in selected_metrics:
                data['num_abnormal_degree_nodes'] = nab

        # — Segments & tortuosity & masks & aggregation —
        if need_segments:
            seg_lists, seg_counts = extract_segments_from_component_using_shortest_path(Gc, dist_map)

            segments_info = []
            agg_curv      = []
            agg_curv2     = []

            for r_idx, seg_list in enumerate(seg_lists):
                seg_metrics = []
                seg_masks   = []
                total_geo   = 0.0
                sum_curv    = 0.0
                sum_curv2   = 0.0

                for seg in seg_list:
                    pts = np.array(seg)
                    sm  = {}

                    # geodesic length
                    L_geo = sum(np.linalg.norm(pts[i]-pts[i+1]) for i in range(len(pts)-1))
                    sm['geodesic_length'] = L_geo
                    total_geo += L_geo

                    # average diameter
                    if 'avg_diameter' in selected_metrics:
                        sm['avg_diameter'] = np.mean([2*dist_map[tuple(p)] for p in pts])

                    # tortuosity
                    counts_arr = np.array([seg_counts[r_idx].get(tuple(p),1) for p in pts])
                    tort = compute_tortuosity_metrics(pts, smoothing=0, n_samples=500, counts=counts_arr)
                    sm.update(tort)

                    if 'spline_mean_curvature' in selected_metrics:
                        sm['spline_mean_curvature'] = tort['spline_mean_curvature']
                        sum_curv  += tort['spline_mean_curvature'] * L_geo
                    if 'spline_mean_square_curvature' in selected_metrics:
                        sm['spline_mean_square_curvature'] = tort['spline_mean_square_curvature']
                        sum_curv2 += tort['spline_mean_square_curvature'] * L_geo

                    # segment mask
                    if save_masks:
                        mask_i = np.zeros_like(skel, dtype=bool)
                        for node in seg:
                            r = dist_map[tuple(node)]
                            if r<=0: continue
                            rr = int(np.ceil(r)); x0,y0,z0=node
                            for dx in range(-rr,rr+1):
                                for dy in range(-rr,rr+1):
                                    for dz in range(-rr,rr+1):
                                        xi,yi,zi = x0+dx,y0+dy,z0+dz
                                        if (0<=xi<mask_i.shape[0] and
                                            0<=yi<mask_i.shape[1] and
                                            0<=zi<mask_i.shape[2] and
                                            dx*dx+dy*dy+dz*dz <= r*r):
                                            mask_i[xi,yi,zi] = True
                        seg_masks.append(nib.Nifti1Image(mask_i.astype(np.uint8), affine, header))

                    seg_metrics.append(sm)

                # aggregate for this root
                if total_geo > 0:
                    agg_curv.append(sum_curv/total_geo)
                    agg_curv2.append(sum_curv2/total_geo)
                else:
                    agg_curv.append(np.nan)
                    agg_curv2.append(np.nan)

                segments_info.append({
                    'segment_metrics': seg_metrics,
                    'segment_masks':   seg_masks
                })

            data['segments_by_root'] = segments_info
            # store aggregated tortuosity in the format save_results expects
            data['aggregated_tortuosity_by_root'] = [
                {'mean_curvature':       agg_curv[i],
                 'mean_square_curvature':agg_curv2[i]}
                for i in range(3)
            ]

        results[cid] = data

    return results


if __name__=='__main__':
    p=argparse.ArgumentParser(description='Vessel analysis; disable segment masks with flag')
    p.add_argument('input',help='Path to vessel mask (.nii/.nii.gz or .npy/.npz)')
    p.add_argument('--min_size',type=int,default=32,help='Minimum cc size')
    p.add_argument('--metrics',nargs='+',default=None,help='Metrics to compute')
    p.add_argument('--output_folder',default='./VESSEL_METRICS',help='Save dir')
    p.add_argument('--no_segment_masks',action='store_true',help='Disable segment mask construction/saving')
    
    args=p.parse_args()
    all_keys = [
        'total_length', 'num_bifurcations', 'bifurcation_density', 'volume',
        'fractal_dimension', 'lacunarity', 'geodesic_length', 'avg_diameter',
        'spline_arc_length', 'spline_chord_length', 'spline_mean_curvature',
        'spline_mean_square_curvature', 'spline_rms_curvature', 'arc_over_chord',
        'fit_rmse', 'num_loops', 'num_abnormal_degree_nodes'
    ]
    selected=set(args.metrics) if args.metrics else set(all_keys)
    invalid=selected-set(all_keys)
    if invalid: raise ValueError(f'Invalid metrics: {invalid}')
    
    save_masks=not args.no_segment_masks
    results=process(args.input,args.min_size,selected,save_masks)
    save_results(results,args.output_folder,save_masks)
    aggregate_segmentation_metrics(os.path.abspath(args.output_folder))
    print(f'Results saved to: {os.path.abspath(args.output_folder)}')

