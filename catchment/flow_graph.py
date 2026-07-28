"""Shared low-level primitives over the D8 flow-direction graph.

Flow direction gives every cell exactly one outgoing edge (or none, if
terminal), and is guaranteed acyclic (see accumulation.py's tie-break rule).
That makes the reversed (upstream) graph a proper forest: every cell has
exactly one path to exactly one root. Several tasks - subbasin delineation,
HAND - are all "propagate some per-root value to everything that drains to
that root," which is why they share one traversal primitive here instead of
each reimplementing it.
"""

import numba
import numpy as np
import xarray as xr


@numba.njit
def _downstream_index_numba(flow_dir_arr, dirmap):
    nrows, ncols = flow_dir_arr.shape
    downstream = np.full(nrows * ncols, -1, dtype=np.int64)
    for row in range(nrows):
        for col in range(ncols):
            code = flow_dir_arr[row, col]
            if code == 0 or code == -1 or code == -2:
                continue
            dr, dc = dirmap[code]
            nr, nc = row + dr, col + dc
            downstream[row * ncols + col] = nr * ncols + nc
    return downstream


@numba.njit
def _propagate_from_seeds_numba(is_seed_flat, seed_values, downstream, nrows, ncols):
    """Multi-source flood fill: propagate each row of `seed_values` from its
    seed cell to every cell that (transitively) drains into it.

    is_seed_flat: bool (n,) - which flat cells are seeds (already resolved)
    seed_values: float64 (k, n) - k value layers; only entries at seed cells
        are meaningful as input, the rest are overwritten if reached
    downstream: int64 (n,) - each cell's downstream flat index, or -1

    Returns (result (k, n) float64, reached (n,) bool). `reached` is False
    (and `result` meaningless) for cells that never drain into any seed -
    same "unassigned" case the old per-reach catchment approach left as 0.
    """
    n = nrows * ncols
    k = seed_values.shape[0]
    result = seed_values.copy()
    reached = is_seed_flat.copy()

    queue = np.empty(n, dtype=np.int64)
    head = 0
    tail = 0
    for idx in range(n):
        if is_seed_flat[idx]:
            queue[tail] = idx
            tail += 1

    dir_drow = np.array([-1, -1, 0, 1, 1, 1, 0, -1], dtype=np.int64)
    dir_dcol = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int64)

    while head < tail:
        idx = queue[head]
        head += 1
        row = idx // ncols
        col = idx % ncols

        for d in range(8):
            nr = row + dir_drow[d]
            nc = col + dir_dcol[d]
            if not (0 <= nr < nrows and 0 <= nc < ncols):
                continue
            n_idx = nr * ncols + nc
            if reached[n_idx]:
                continue
            if downstream[n_idx] == idx:
                for layer in range(k):
                    result[layer, n_idx] = result[layer, idx]
                reached[n_idx] = True
                queue[tail] = n_idx
                tail += 1

    return result, reached


def propagate_downstream(
    is_seed: np.ndarray,
    seed_value_arrays: list,
    flow_directions: xr.DataArray,
    dirmap,
) -> tuple:
    """Python-level convenience around `_propagate_from_seeds_numba`.

    is_seed: 2D bool array - which pixels are seeds.
    seed_value_arrays: list of 2D arrays (same shape as is_seed), one per
        value to propagate - all propagated together in a single traversal.
    flow_directions: flow direction raster (ESRI D8 encoding).
    dirmap: numba-typed ESRI dirmap, e.g. from `_make_numba_esri_dirmap()`.

    Returns (list of 2D propagated arrays matching input order, 2D bool
    'reached' mask).
    """
    nrows, ncols = is_seed.shape
    downstream = _downstream_index_numba(flow_directions.values, dirmap)

    seed_values = np.stack(
        [arr.astype(np.float64).reshape(-1) for arr in seed_value_arrays]
    )
    is_seed_flat = is_seed.reshape(-1)

    result, reached_flat = _propagate_from_seeds_numba(
        is_seed_flat, seed_values, downstream, nrows, ncols
    )

    results = [result[i].reshape(nrows, ncols) for i in range(len(seed_value_arrays))]
    reached = reached_flat.reshape(nrows, ncols)
    return results, reached
