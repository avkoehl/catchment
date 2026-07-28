import numpy as np


def group_pixels_by_id(id_raster: np.ndarray) -> dict:
    """Group pixel (row, col) coordinates by integer raster value, in one pass.

    Equivalent to building ``{id: np.where(id_raster == id)}`` for every unique
    id in the raster, but O(pixels log pixels) instead of O(pixels * n_ids).
    Groups preserve row-major (np.where) order within each id. Zero and
    non-finite values (nodata/NaN) are treated as background and excluded.
    """
    valid = np.isfinite(id_raster) & (id_raster != 0)
    rows, cols = np.where(valid)
    ids = id_raster[rows, cols].astype(np.int64)

    order = np.argsort(ids, kind="stable")
    ids_sorted = ids[order]
    rows_sorted = rows[order]
    cols_sorted = cols[order]

    unique_ids, start_idx = np.unique(ids_sorted, return_index=True)
    end_idx = np.append(start_idx[1:], len(ids_sorted))

    return {
        int(uid): (rows_sorted[s:e], cols_sorted[s:e])
        for uid, s, e in zip(unique_ids, start_idx, end_idx)
    }
