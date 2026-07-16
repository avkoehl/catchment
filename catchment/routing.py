import numba
import numpy as np
import xarray as xr

from .dirmap import _make_numba_esri_dirmap


def route_stream(
    stream_mask: xr.DataArray,
    flow_directions: xr.DataArray,
    flow_accumulation: xr.DataArray,
    add_downstream_cell: bool = False,
) -> list[tuple[int, int]]:
    """
    Given a mask of a single stream segment, trace the path from the start to
    the end. Uses flow accumulation to find the first and last cells in the
    segment. Follows flow directions to trace the path.

    Args:
        stream_mask: array with non-zero values indicating the stream segment to trace.
        flow_directions: array of flow directions (ESRI style).
        flow_accumulation: array of flow accumulation values.
    Returns:
        List of (row, col) tuples representing the traced path.
    """

    dirmap = _make_numba_esri_dirmap()
    start, end = _determine_start_and_end(stream_mask, flow_accumulation)
    break_conditions_arr = stream_mask.data <= 0
    path = _path_numba(
        start[0], start[1], flow_directions.data, dirmap, break_conditions_arr
    )
    if path[0] != start or path[-1] != end:
        raise ValueError("Traced path does not match start and end points")

    stream_cells = np.argwhere(stream_mask.data)
    stream_cells_set = set((row, col) for row, col in stream_cells)
    path_set = set(path)
    if stream_cells_set != path_set:
        raise ValueError("Traced path does not cover all stream cells")

    if not add_downstream_cell:
        return path

    # if the final cell points somewhere else, add that cell to the path
    final_direction = flow_directions.data[path[-1][0], path[-1][1]]
    if final_direction not in (-1, -2, 0):
        drow, dcol = dirmap[final_direction]
        next_row = path[-1][0] + drow
        next_col = path[-1][1] + dcol
        if (
            0 <= next_row < flow_directions.shape[0]
            and 0 <= next_col < flow_directions.shape[1]
        ):
            path.append((next_row, next_col))
    return path


@numba.njit
def _path_numba(row, col, flow_directions_arr, dirmap, break_conditions_arr):
    """Trace the path from a starting cell until a break condition or outlet/pit is met"""
    nrows, ncols = flow_directions_arr.shape
    path = [(row, col)]

    while True:
        current_direction = flow_directions_arr[row, col]
        if current_direction in (-1, -2, 0):
            break

        drow, dcol = dirmap[current_direction]
        next_row = row + drow
        next_col = col + dcol

        if not (0 <= next_row < nrows and 0 <= next_col < ncols):
            break

        if break_conditions_arr[next_row, next_col] == 1:
            break

        path.append((next_row, next_col))
        row, col = next_row, next_col

    return path


def _determine_start_and_end(stream_mask, flow_acc):
    """Given a mask of a single stream segment, determine the start and end points"""
    # find the stream cells with the lowest and highest flow accumulation
    stream_cells = np.argwhere(stream_mask.data)
    flow_acc_values = flow_acc.data[stream_mask.data]
    min_idx = np.argmin(flow_acc_values)
    max_idx = np.argmax(flow_acc_values)
    start = tuple(stream_cells[min_idx])
    end = tuple(stream_cells[max_idx])
    return start, end
