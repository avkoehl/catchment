"""
Code for finding detrending the down valley trend of a DEM using a linestring and IDW
modified from: https://github.com/DahnJ/REM-xarray
"""

import numpy as np
from scipy.spatial import cKDTree as KDTree
from scipy.sparse.csgraph import dijkstra
import xarray as xr

from .cost import _create_cost_graph
from .dirmap import _make_numba_esri_dirmap
from .flow_graph import propagate_downstream


def compute_hand(
    dem: xr.DataArray,
    streams: xr.DataArray,
    flow_directions: xr.DataArray,
) -> xr.DataArray:
    """
    Compute Height Above Nearest Drainage (HAND): for each pixel, its
    elevation minus the elevation of the nearest stream pixel it drains to.

    Multi-source flood fill over the reversed flow-direction graph (see
    `flow_graph.py`), seeded with each stream pixel's own elevation and
    propagated upstream - the same traversal `delineate_subbasins` uses,
    just propagating elevation instead of reach id. Pixels that never drain
    to any stream cell get NaN (matching the "unassigned" case elsewhere).

    Args:
        dem: Digital Elevation Model (conditioned).
        streams: Stream network raster (0 for non-stream pixels).
        flow_directions: Flow direction raster (ESRI D8 encoding).
    Returns:
        xr.DataArray: HAND values for each cell in the DEM.
    """
    stream_values = streams.values
    if np.issubdtype(stream_values.dtype, np.floating):
        is_seed = np.isfinite(stream_values) & (stream_values != 0)
    else:
        is_seed = stream_values != 0

    elevation = dem.values.astype(np.float64)

    dirmap = _make_numba_esri_dirmap()
    [nearest_stream_elev], reached = propagate_downstream(
        is_seed, [elevation], flow_directions, dirmap
    )

    hand_arr = elevation - nearest_stream_elev
    hand_arr[~reached] = np.nan
    hand_arr[np.isnan(elevation)] = np.nan

    hand = dem.copy(data=hand_arr)
    hand.encoding = {}
    hand = hand.rio.write_nodata(np.nan)
    return hand


def compute_rem(
    dem: xr.DataArray,
    stream_mask: xr.DataArray,
    k: int = 5,
    fit_regression: bool = False,
    dist_method: str = "cost-distance",
) -> xr.DataArray:
    """
    Compute Relative Elevation Model (REM) by detrending the down-valley trend of a DEM using a stream mask and IDW interpolation.
    Args:
        dem (xr.DataArray): Input DEM.
        stream_mask (xr.DataArray): Binary mask of stream locations.
        k (int, optional): Number of nearest stream points to use for IDW. Defaults to 5.
        fit_regression (bool, optional): Whether to fit a regression to the stream elevations before IDW. Defaults to False.
        dist_method (str, optional): Method to compute distances. Must be 'euclidean' or 'cost-distance'. Defaults to 'cost-distance'.
    Returns:
        xr.DataArray: Relative Elevation Model (REM) with the down-valley trend removed.
    """
    if dist_method == "euclidean":
        dists, inds = find_nearest_stream_pts_euclidean(dem, stream_mask, k)
    elif dist_method == "cost-distance":
        cost_graph = _create_cost_graph(dem)
        dists, inds = find_nearest_stream_pts_cost_distance(
            dem, stream_mask, cost_graph, k
        )
    else:
        raise ValueError("dist_method must be 'euclidean' or 'cost-distance'")

    stream_rows, stream_cols = np.where(stream_mask.values)
    stream_elevations = dem.values[stream_rows, stream_cols]
    if fit_regression:
        stream_elevations = fit_stream_elevations(stream_elevations)

    stream_surface = idw_stream_surface(stream_elevations, dists, inds, power=1)

    rem = dem - stream_surface
    return rem


def find_nearest_stream_pts_euclidean(dem, stream_mask, k):
    stream_rows, stream_cols = np.where(stream_mask.values)

    stream_x = dem.x.values[stream_cols]
    stream_y = dem.y.values[stream_rows]
    stream_spatial_coords = np.column_stack((stream_x, stream_y))

    tree = KDTree(stream_spatial_coords)

    dem_x, dem_y = np.meshgrid(dem.x.values, dem.y.values)
    dem_spatial_coords = np.column_stack((dem_x.ravel(), dem_y.ravel()))

    rows, cols = dem.shape
    distances, indices = tree.query(dem_spatial_coords, k=k)
    distances = distances.reshape(rows, cols, k)
    indices = indices.reshape(rows, cols, k)
    return distances, indices


def find_nearest_stream_pts_cost_distance(dem, stream_mask, cost_graph, k):
    stream_rows, stream_cols = np.where(stream_mask.values)
    stream_flat_indices = np.ravel_multi_index((stream_rows, stream_cols), dem.shape)

    distances = dijkstra(
        csgraph=cost_graph,
        directed=True,
        indices=stream_flat_indices,
        return_predecessors=False,
    )  # shape (n_stream_points, rows*cols)
    distances = distances.T  # shape (rows*cols, n_stream_points)
    knn_indices_flat = np.argpartition(distances, k - 1, axis=1)[
        :, :k
    ]  # shape: (rows*cols, k)
    row_indices = np.arange(distances.shape[0])[:, None]  # shape: (rows*cols, 1)
    distances_flat = distances[row_indices, knn_indices_flat]  # shape: (rows*cols, k)
    rows, cols = dem.shape
    distances = distances_flat.reshape(rows, cols, k)
    indices = knn_indices_flat.reshape(rows, cols, k)
    return distances, indices


def fit_stream_elevations(elevations):
    x = np.arange(len(elevations))
    coeffs = np.polyfit(x, elevations, 2)  # 2nd order polynomial
    return np.polyval(coeffs, x)


def idw_stream_surface(stream_elevations, distances, indices, power):
    weights = 1 / (distances**power + 1e-12)

    weights_sum = np.sum(weights, axis=-1, keepdims=True)
    weights_normalized = weights / weights_sum

    interpolated_values = np.sum(
        weights_normalized * stream_elevations[indices], axis=-1
    )

    return interpolated_values
