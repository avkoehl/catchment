import numpy as np
import xarray as xr

from .dirmap import _make_numba_esri_dirmap
from .nodes import _find_stream_nodes


def compute_distance_to_head(
    streams: xr.DataArray, flow_direction: xr.DataArray
) -> xr.DataArray:
    """
    For each cell in the stream raster, compute the maximum upstream length
    (distance to furthest headwater)

    Args:
        streams: A binary raster where stream cells are 1 and non-stream cells are 0.
        flow_direction: A raster representing flow direction using ESRI convention.
    Returns:
        A raster where each stream cell contains the maximum upstream length in map units.
    """
    dirmap = _make_numba_esri_dirmap()
    sources, _, _ = _find_stream_nodes(streams, flow_direction)
    distance_arr = _distance_from_head(
        streams.data, sources, flow_direction.data, dirmap
    )
    distance_raster = flow_direction.copy(data=distance_arr)
    distance_raster *= np.abs(flow_direction.rio.resolution()[0])
    return distance_raster


def _distance_from_head(stream_arr, headwater_points, flow_dir_arr, dirmap):
    nrows, ncols = flow_dir_arr.shape
    distance_arr = np.zeros((nrows, ncols), dtype=np.float32)

    for point in headwater_points:
        row, col = point
        distance = 0.0

        while True:
            if distance >= distance_arr[row, col]:
                distance_arr[row, col] = distance
            else:
                break

            current_direction = flow_dir_arr[row, col]
            if current_direction in (-1, -2, 0):
                break

            drow, dcol = dirmap[current_direction]
            next_row = row + drow
            next_col = col + dcol

            if not (0 <= next_row < nrows and 0 <= next_col < ncols):
                break

            if stream_arr[next_row, next_col] == 0:
                break

            # Assuming each cell is 1 unit length; modify if cell size is different
            dist_increment = np.sqrt(drow**2 + dcol**2)
            distance += dist_increment
            row, col = next_row, next_col
    return distance_arr
