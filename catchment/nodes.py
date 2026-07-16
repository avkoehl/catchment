import numba
import numpy as np
import xarray as xr

from .dirmap import _make_numba_esri_dirmap


def _find_stream_nodes(
    stream_raster: xr.DataArray, flow_directions: xr.DataArray
) -> tuple:
    """Identify source points and confluence points in a stream network
    Args:
        stream_raster: Raster representing the stream network (non-zero values indicate streams)
        flow_directions: Raster representing flow directions using ESRI convention
    Returns:
        Tuple containing lists of source points, confluence points, and outlet points
    """

    dirmap = _make_numba_esri_dirmap()
    sources, confluences, outlets = _find_stream_nodes_numba(
        stream_raster.data, flow_directions.data, dirmap
    )
    return sources, confluences, outlets


@numba.njit
def _find_stream_nodes_numba(stream_arr, flow_directions_arr, dirmap):
    """Find source points (headwaters) and confluence points in stream network"""
    nrows, ncols = flow_directions_arr.shape
    inflow_count = np.zeros((nrows, ncols), dtype=np.uint8)

    # Count how many stream cells flow into each cell
    for row in range(nrows):
        for col in range(ncols):
            if stream_arr[row, col] == 0:
                continue

            current_direction = flow_directions_arr[row, col]
            if current_direction in (-1, -2, 0):
                continue

            drow, dcol = dirmap[current_direction]
            next_row = row + drow
            next_col = col + dcol

            if 0 <= next_row < nrows and 0 <= next_col < ncols:
                if stream_arr[next_row, next_col] != 0:
                    inflow_count[next_row, next_col] += 1

    # Find source points (no inflow) and confluence points (multiple inflows)
    sources = []
    confluences = []

    for row in range(nrows):
        for col in range(ncols):
            if stream_arr[row, col] == 0:
                continue

            if inflow_count[row, col] == 0:
                sources.append((row, col))
            elif inflow_count[row, col] > 1:
                confluences.append((row, col))

    # Find outlet points (no outflow)
    outlets = []
    for row in range(nrows):
        for col in range(ncols):
            if stream_arr[row, col] == 0:
                continue

            current_direction = flow_directions_arr[row, col]
            if current_direction in (-1, -2, 0):
                outlets.append((row, col))

    return sources, confluences, outlets
