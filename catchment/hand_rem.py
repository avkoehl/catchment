"""
Code for finding detrending the down valley trend of a DEM using a linestring and IDW
modified from: https://github.com/DahnJ/REM-xarray
"""

import tempfile
import os
import shutil

import rioxarray as rxr
import whitebox
import numpy as np
from scipy.spatial import cKDTree as KDTree
from scipy.sparse.csgraph import dijkstra
import xarray as xr

from .cost import _create_cost_graph
from .adapters import to_pysheds, from_pysheds


def compute_hand_wbt(conditioned_dem, streams):
    working_dir = tempfile.mkdtemp()
    wbt = whitebox.WhiteboxTools()
    wbt.set_working_dir(working_dir)
    wbt.verbose = False

    dem_path = os.path.join(working_dir, "conditioned_dem.tif")
    streams_path = os.path.join(working_dir, "streams.tif")
    hand_path = os.path.join(working_dir, "hand.tif")
    conditioned_dem.rio.to_raster(dem_path)
    streams.rio.to_raster(streams_path)

    wbt.elevation_above_stream(dem_path, streams_path, hand_path)

    if not os.path.exists(hand_path) or os.path.getsize(hand_path) == 0:
        raise FileNotFoundError("WhiteboxTools failed to produce HAND output")

    hand = rxr.open_rasterio(hand_path, masked=True).squeeze().load()
    hand = hand.rio.reproject_match(conditioned_dem)
    shutil.rmtree(working_dir, ignore_errors=True)

    return hand


def compute_hand(
    dem: xr.DataArray,
    streams: xr.DataArray,
    method: str = "wbt",
    flow_directions: None | xr.DataArray = None,
) -> xr.DataArray:
    """
    Compute Height Above Nearest Drainage (HAND) using either WhiteboxTools or PySheds.

    Args:
        dem (xr.DataArray): Digital Elevation Model.
        streams (xr.DataArray): Binary mask of stream locations.
        method (str, optional): Method to compute HAND. Must be 'wbt' or 'pysheds'. Defaults to 'wbt'.
        flow_directions (xr.DataArray, optional): Flow direction raster in D8 encoding, required if method is 'pysheds'.
    Returns:
        xr.DataArray: HAND values for each cell in the DEM.


    """
    if method == "wbt":
        return compute_hand_wbt(dem, streams)
    if method != "pysheds":
        raise ValueError("method must be 'wbt' or 'pysheds'")

    # note ESRI d8 flow direction encoding
    if flow_directions is None:
        raise ValueError("flow_directions must be provided when method is 'pysheds'")

    pysheds_dem, grid = to_pysheds(dem)
    flow_directions, _ = to_pysheds(flow_directions)
    streams, _ = to_pysheds(streams)
    dirmap = (64, 128, 1, 2, 32, 16, 8, 4)
    hand = grid.compute_hand(flow_directions, pysheds_dem, streams > 0, dirmap=dirmap)
    hand = from_pysheds(hand)
    hand = hand.rio.reproject_match(dem)
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
