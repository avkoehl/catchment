import numba
import numpy as np
import xarray as xr

from .dirmap import _make_numba_esri_dirmap


def route_stream(
    rows: np.ndarray,
    cols: np.ndarray,
    label_raster: np.ndarray,
    target_id,
    flow_directions: xr.DataArray,
    flow_accumulation: xr.DataArray,
    add_downstream_cell: bool = False,
) -> list[tuple[int, int]]:
    """
    Given the pixel coordinates of a single stream segment, trace the path
    from the start to the end. Uses flow accumulation to find the first and
    last cells in the segment. Follows flow directions to trace the path.

    Args:
        rows, cols: row/col coordinates of every pixel belonging to the
            stream segment (e.g. from `raster_index.group_pixels_by_id`).
        label_raster: full raster where each stream segment has a unique id
            (used to detect when the trace has left this segment, without
            re-deriving a mask over the whole raster per call).
        target_id: this segment's id in `label_raster`.
        flow_directions: array of flow directions (ESRI style).
        flow_accumulation: array of flow accumulation values.
    Returns:
        List of (row, col) tuples representing the traced path.
    """

    dirmap = _make_numba_esri_dirmap()
    start, end = _determine_start_and_end(rows, cols, flow_accumulation.data)
    path = _path_numba(
        start[0], start[1], flow_directions.data, dirmap, label_raster, target_id
    )
    if path[0] != start or path[-1] != end:
        raise ValueError("Traced path does not match start and end points")

    stream_cells_set = set(zip(rows.tolist(), cols.tolist()))
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
def _path_numba(row, col, flow_directions_arr, dirmap, label_arr, target_id):
    """Trace the path from a starting cell until it leaves target_id's region
    (per label_arr), an outlet/pit is met, or the walk revisits a cell.

    Flow direction from a hydrologically-conditioned DEM is normally acyclic,
    but conditioning (fix_flats on complex flats) or a reprojection step can
    leave a local cycle. Without a visited check, such a cycle would make this
    loop run forever; the visited set is bounded by path length (not raster
    size), so it doesn't reintroduce a per-call full-raster cost.
    """
    nrows, ncols = flow_directions_arr.shape
    path = [(row, col)]
    visited = {row * ncols + col}

    while True:
        current_direction = flow_directions_arr[row, col]
        if current_direction in (-1, -2, 0):
            break

        drow, dcol = dirmap[current_direction]
        next_row = row + drow
        next_col = col + dcol

        if not (0 <= next_row < nrows and 0 <= next_col < ncols):
            break

        if label_arr[next_row, next_col] != target_id:
            break

        next_key = next_row * ncols + next_col
        if next_key in visited:
            break

        path.append((next_row, next_col))
        visited.add(next_key)
        row, col = next_row, next_col

    return path


def _determine_start_and_end(rows, cols, flow_acc_data):
    """Given the pixel coordinates of a single stream segment, determine the
    start (lowest flow accumulation) and end (highest) points"""
    flow_acc_values = flow_acc_data[rows, cols]
    min_idx = np.argmin(flow_acc_values)
    max_idx = np.argmax(flow_acc_values)
    start = (int(rows[min_idx]), int(cols[min_idx]))
    end = (int(rows[max_idx]), int(cols[max_idx]))
    return start, end
