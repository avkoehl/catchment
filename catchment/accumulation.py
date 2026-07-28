import tempfile
import os
import time
import shutil

import numba
import numpy as np
import rioxarray as rxr
import xarray as xr
import whitebox

from .dirmap import _make_numba_esri_dirmap
from .flow_graph import _downstream_index_numba


def condition_dem(dem, max_retries=3, wait_time=1.0):
    """Condition DEM using WhiteboxTools fill_depressions with retry logic."""
    for attempt in range(1, max_retries + 1):
        try:
            # Create fresh temp dir and WBT instance
            working_dir = tempfile.mkdtemp()
            wbt = whitebox.WhiteboxTools()
            wbt.set_working_dir(working_dir)
            wbt.verbose = False

            # Write DEM to disk
            dem_path = os.path.join(working_dir, "dem.tif")
            filled_path = os.path.join(working_dir, "filled_dem.tif")
            dem.rio.to_raster(dem_path)

            # Run fill depressions
            wbt.fill_depressions(dem_path, filled_path, fix_flats=True)

            # Check for output
            if not os.path.exists(filled_path) or os.path.getsize(filled_path) == 0:
                raise FileNotFoundError("WhiteboxTools failed to produce output")

            # Load result
            conditioned_dem = rxr.open_rasterio(filled_path, masked=True).squeeze().load()
            shutil.rmtree(working_dir, ignore_errors=True)
            return conditioned_dem

        except Exception as e:
            print(f"[Attempt {attempt}/{max_retries}] fill_depressions failed: {e}")
            if attempt < max_retries:
                time.sleep(wait_time)
                continue
            else:
                raise RuntimeError(
                    f"condition_dem failed after {max_retries} attempts"
                ) from e


def flow_accumulation_workflow(
    dem: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Given a DEM, compute the conditioned DEM, flow directions, and flow
    accumulation. Depression filling/flat resolution is done with
    WhiteboxTools; D8 flow direction and accumulation are computed directly
    on the conditioned DEM's own grid (ESRI flow direction encoding).

    Args:
        dem: DEM raster
    Returns:
        (conditioned DEM, flow directions, and flow accumulation)

    """
    conditioned_dem = condition_dem(dem)
    # align once, before deriving flow_dir/flow_acc, so those two (computed
    # directly on this grid) never need their own reprojection afterward
    conditioned_dem = conditioned_dem.rio.reproject_match(dem)

    flow_directions = compute_flow_directions(conditioned_dem)
    flow_accumulation = compute_flow_accumulation(conditioned_dem, flow_directions)

    return conditioned_dem, flow_directions, flow_accumulation


# ---------------------------------------------------------------------------
# D8 flow direction
# ---------------------------------------------------------------------------

# ESRI direction codes, in the same order/encoding as dirmap.py
_DIR_CODES = np.array([64, 128, 1, 2, 4, 8, 16, 32], dtype=np.int16)
_DIR_DROW = np.array([-1, -1, 0, 1, 1, 1, 0, -1], dtype=np.int64)
_DIR_DCOL = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int64)


def compute_flow_directions(dem: xr.DataArray) -> xr.DataArray:
    """Compute D8 flow direction from a conditioned DEM (ESRI encoding).

    Steepest-descent, with a tie-break rule chosen specifically to make the
    result acyclic by construction: exact elevation ties (flats) only ever
    flow toward a strictly larger row-major index, never a smaller one. Since
    every step is either strictly-decreasing elevation or flat-with-
    increasing-index, no sequence of steps can return to its starting cell.
    """
    res_x, res_y = dem.rio.resolution()
    dx, dy = abs(res_x), abs(res_y)
    flow_dir_arr = _flow_direction_numba(
        dem.values, dx, dy, _DIR_CODES, _DIR_DROW, _DIR_DCOL
    )
    flow_dir = dem.copy(data=flow_dir_arr)
    flow_dir.encoding = {}  # drop dem's encoding (e.g. a stale _FillValue from
    # the conditioned DEM's source file) so it doesn't leak into this raster
    # or ones derived from it (e.g. stream_network via flow_dir.copy(data=...))
    flow_dir = flow_dir.rio.write_nodata(0)
    return flow_dir


@numba.njit
def _flow_direction_numba(dem_arr, dx, dy, dir_codes, dir_drow, dir_dcol):
    nrows, ncols = dem_arr.shape
    flow_dir = np.zeros((nrows, ncols), dtype=np.int16)

    diag = np.sqrt(dx * dx + dy * dy)
    dists = np.array([dy, diag, dx, diag, dy, diag, dx, diag])

    for row in range(nrows):
        for col in range(ncols):
            elev = dem_arr[row, col]
            if np.isnan(elev):
                continue

            self_index = row * ncols + col
            best_slope = 0.0
            best_code = 0
            best_flat_code = 0
            best_flat_index = -1

            for k in range(8):
                nr = row + dir_drow[k]
                nc = col + dir_dcol[k]
                if not (0 <= nr < nrows and 0 <= nc < ncols):
                    continue
                nelev = dem_arr[nr, nc]
                if np.isnan(nelev):
                    continue

                if nelev < elev:
                    slope = (elev - nelev) / dists[k]
                    if slope > best_slope:
                        best_slope = slope
                        best_code = dir_codes[k]
                elif nelev == elev:
                    # flat: only allowed to flow toward strictly larger
                    # row-major index, so flats can never cycle
                    n_index = nr * ncols + nc
                    if n_index > self_index and n_index > best_flat_index:
                        best_flat_index = n_index
                        best_flat_code = dir_codes[k]

            if best_code != 0:
                flow_dir[row, col] = best_code
            elif best_flat_code != 0:
                flow_dir[row, col] = best_flat_code
            # else: leave as 0 (terminal - pit/edge/no valid direction)

    return flow_dir


# ---------------------------------------------------------------------------
# Flow accumulation
# ---------------------------------------------------------------------------


def compute_flow_accumulation(
    dem: xr.DataArray, flow_directions: xr.DataArray
) -> xr.DataArray:
    """Compute D8 flow accumulation (contributing cell count) from a
    conditioned DEM and its flow directions.

    Processes cells in a single topological pass: descending elevation
    primarily, ascending row-major index as a tiebreaker (matching the
    flat-tie rule in `compute_flow_directions`, so any cell that flows into
    another is guaranteed to be processed first). O(N log N) for the sort,
    O(N) for the accumulation push - no sparse solver or graph needed, since
    flow direction is a single-parent forest (at most one outgoing edge per
    cell), which is exactly the structure a topological push exploits directly.
    """
    dem_arr = dem.values
    nrows, ncols = dem_arr.shape
    valid = ~np.isnan(dem_arr)

    flat_index = np.arange(nrows * ncols, dtype=np.int64)
    valid_idx = flat_index[valid.ravel()]
    valid_elev = dem_arr.ravel()[valid_idx]

    order = valid_idx[np.lexsort((valid_idx, -valid_elev))]

    dirmap = _make_numba_esri_dirmap()
    downstream = _downstream_index_numba(flow_directions.values, dirmap)

    flow_acc = np.zeros(nrows * ncols, dtype=np.int64)
    flow_acc[valid_idx] = 1
    flow_acc = _push_accumulation_numba(order, downstream, flow_acc)
    flow_acc = flow_acc.reshape(nrows, ncols).astype(np.float64)
    flow_acc[~valid] = np.nan

    result = dem.copy(data=flow_acc)
    result.encoding = {}  # see compute_flow_directions - drop stale encoding
    result = result.rio.write_nodata(np.nan)
    return result


@numba.njit
def _push_accumulation_numba(order, downstream, flow_acc):
    for i in range(len(order)):
        idx = order[i]
        target = downstream[idx]
        if target != -1:
            flow_acc[target] += flow_acc[idx]
    return flow_acc
