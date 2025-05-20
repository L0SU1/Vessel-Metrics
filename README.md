## Vessel Analysis

A command-line tool to analyze 3D vessel masks by skeletonizing them, building a graph representation, extracting vessel segments, and computing a variety of geometric and tortuosity metrics.

---

## Overview

`VESSEL_METRICS.py` processes an input 3D vessel mask through the following main stages:

1. **Loading & Preprocessing**
2. **Skeletonization & Distance Mapping**
3. **Graph Construction & Pruning**
4. **Connected Component Analysis**
5. **General Metrics Computation**
6. **Structural Metrics Computation**
7. **Segment Extraction & Tortuosity Analysis**
8. **Results Saving & Aggregation**

Each stage is described in detail below.

---

## 1. Loading & Preprocessing

* **Input formats**: NIfTI (`.nii`, `.nii.gz`) or NumPy (`.npy`, `.npz`).
* **Thresholding**: Voxel values > 0 are considered foreground.
* **Small object removal**: Uses `skimage.morphology.remove_small_objects` to eliminate connected components smaller than `--min_size` voxels (default 32).

```python
arr = mask_data > 0
clean = remove_small_objects(arr, min_size)
```

---

## 2. Skeletonization & Distance Mapping

* **Skeletonize**: Computes a 3D medial axis of the cleaned mask via `skimage.morphology.skeletonize`.
* **Distance transform**: Computes Euclidean distance from each foreground voxel to the nearest background using `scipy.ndimage.distance_transform_edt`. The resulting `dist_map` provides per-voxel radius estimates.

```python
skel = skeletonize(clean)
dist_map = distance_transform_edt(clean)
```

---

## 3. Graph Construction & Pruning

### 3.1 Build Graph

* Each skeleton voxel becomes a graph node at its (x, y, z) coordinate.
* Edges connect 6-neighbors (faces) and also diagonal neighbors (within one voxel) for full connectivity making it 26-connectivity.
* Edge **weight** = Euclidean distance between voxel centers (1.0 for face neighbors, √2 or √3 for diagonals).

```python
def build_graph(skel):
    for each voxel in skel:
        for neighbor in 26-neighborhood:
            if neighbor is also skeleton:
                G.add_edge(voxel, neighbor, weight=distance)
```

### 3.2 Prune Triangles

* Detects all simple cycles of length 3 (triangles) via `networkx.cycle_basis`. They appear in occurrence of the bifurctions for how 26-connectivity build edges.
* In each triangle, removes the heaviest edge (largest weight) to eliminate spurious loops. 
```python
for cycle in cycle_basis(G):
    if len(cycle)==3:
        remove heaviest edge
```

---

## 4. Connected Component Analysis

* Splits the pruned graph `G` into connected components (`networkx.connected_components`).
* For each component:

  * **Reconstruct mask**: Grows spheres at each skeleton node using `dist_map` radii, reassembling the original vessel thickness.
  * Stores this as a NIfTI image (`nibabel.Nifti1Image`).

```python
for node in Gc.nodes():
    r = dist_map[node]
    paint sphere of radius r
```

---

## 5. General Metrics Computation

For each component, computes:

* **Total length**: Sum of all edge weights in the component.
* **Number of bifurcations**: Nodes with degree ≥ 3.
* **Bifurcation density**: (# of bifurcations) / (total length).
* **Volume**: Approximates each edge as a cylinder: π·(avg\_radius)²·length.
* **Fractal dimension**: 3D box-counting on the set of skeleton coordinates using `sklearn.linear_model.LinearRegression` on log(counts) vs log(1/box\_size).
* **Lacunarity**: Variance-to-mean² of point counts in a grid of boxes covering the component.

```python
num_bif = sum(degree>=3)
total_len = sum(weights)
volume = sum(pi*r_avg^2*length)
```

---

## 6. Structural Metrics Computation

Calculates:

* **Number of loops**: Count of independent cycles via `len(cycle_basis(Gc))`.
* **Abnormal nodes**: Nodes with degree > 3.

---

## 7. Segment Extraction & Tortuosity Analysis

### 7.1 Root Selection

Identifies three roots per component:

1. **Largest endpoint**: Skeleton endpoint (degree==1) with max diameter (2·radius).
2. **Second-largest endpoint**.
3. **Largest bifurcation**: Node with degree ≥ 3 and max diameter.

### 7.2 Shortest-path segments

For each root, computes shortest paths to all other endpoints using `networkx.shortest_path` (Dijkstra on weights).

### 7.3 Tortuosity metrics per segment

For each segment path of voxel coords:

* **Geodesic length**: Sum of Euclidean distances along the path.
* **Average diameter**: Mean of 2·dist\_map at each point.
* **Spline-based tortuosity**: Fits a cubic B-spline to the 3D points (`scipy.interpolate.splprep`), reparametrizes by arc length, then computes:

  * **Arc length** & **Chord length**
  * **Curvature** κ(s) = ‖r'(s)×r''(s)‖ / ‖r'(s)‖³
  * **Weighted integrals**: ∫κ(s)/n(s) ds and ∫\[κ(s)/n(s)]² ds, where n(s) are per-point counts to down-weight heavily-traveled points.
  * **RMS curvature** and **fit RMSE**.

```python
tck,u = splprep(pts.T)
evaluate derivs, compute curvature, trapz integrals
```

### 7.4 Aggregation

* **Per-root aggregated curvature**: length-weighted mean of each root’s segment curvatures:

  $C_root = \frac{\sum_{seg}(mean_curv_seg \times L_geo_seg)}{\sum_{seg} L_geo_seg}$

Stored alongside general metrics.

---

## 8. Results Saving & Aggregation

### Per-component

Under `<output_folder>/Conn_comp_<i>/`:

* `General_metrics.csv` containing all requested general and aggregated tortuosity metrics.
* `<i>_skeleton.nii.gz`: Reconstructed skeleton mask.
* `Segments/`: Subfolders for each root, each containing per-segment `Segment_metrics.csv` and optional `Segment.nii.gz` masks.

### Whole-mask

After all components are saved, runs `aggregate_segmentation_metrics`, which:

* Reads each component’s `General_metrics.csv`
* Sums general fields across components
* Computes overall length-weighted averages for each of the six tortuosity metrics, **only** including components where those values exist
* Writes `segmentation_metrics.csv` in `<output_folder>` with all sums, averages, `grand_total_length`, and `num_components`.

---

## Quick Start

```bash
python VESSEL_METRICS.py my_vessels.nii.gz \
    --min_size 64 \
    --metrics total_length volume num_loops spline_mean_curvature \
    --output_folder ./results \
    --no_segment_masks
```

This produces:

```
./results/Conn_comp_1/...
./results/Conn_comp_2/...
./results/segmentation_metrics.csv
```

---
