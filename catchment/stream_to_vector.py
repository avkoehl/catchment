import geopandas as gpd
import numpy as np
from rasterio.transform import xy
from shapely.geometry import LineString
import xarray as xr

from .routing import _path_numba
from .dirmap import _make_numba_esri_dirmap
from .raster_index import group_pixels_by_id

_ESRI_DIRMAP = {
    64: (-1, 0),  # North
    128: (-1, 1),  # Northeast
    1: (0, 1),  # East
    2: (1, 1),  # Southeast
    4: (1, 0),  # Southeast
    8: (1, -1),  # Southwest
    16: (0, -1),  # West
    32: (-1, -1),  # Northwest
    -1: (0, 0),  # outlet/terminal
    -2: (0, 0),
    0: (0, 0),
}


def streams_to_vector(
    stream_raster: xr.DataArray,
    flow_directions: xr.DataArray,
    flow_accumulation: xr.DataArray,
) -> gpd.GeoDataFrame:
    raster_vals = stream_raster.values
    fa_vals = flow_accumulation.values
    fd_vals = flow_directions.values
    transform = stream_raster.rio.transform()
    crs = stream_raster.rio.crs
    shape = raster_vals.shape

    # single pass: group pixel indices by stream_id
    pixel_lookup = group_pixels_by_id(raster_vals)

    flowlines = []
    for sid, (rows, cols) in pixel_lookup.items():
        line = _vectorize_single_stream(
            sid, rows, cols, fa_vals, fd_vals, raster_vals, shape, transform
        )
        if line is not None:
            flowlines.append({"geometry": line, "stream_id": int(sid)})

    return gpd.GeoDataFrame(flowlines, crs=crs)


def _vectorize_single_stream(sid, rows, cols, fa_vals, fd_vals, raster_vals, shape, transform):
    fa_at_cells = fa_vals[rows, cols]
    start = (int(rows[np.argmin(fa_at_cells)]), int(cols[np.argmin(fa_at_cells)]))
    end = (int(rows[np.argmax(fa_at_cells)]), int(cols[np.argmax(fa_at_cells)]))

    dirmap = _make_numba_esri_dirmap()
    path = _path_numba(start[0], start[1], fd_vals, dirmap, raster_vals, sid)

    if len(path) < 2:
        return None

    if path[0] != start or path[-1] != end:
        raise ValueError(f"Traced path start/end mismatch for stream_id {sid}")

    path_set = set(path)
    stream_cells_set = set(zip(rows.tolist(), cols.tolist()))
    if path_set != stream_cells_set:
        raise ValueError(
            f"Traced path does not cover all stream cells for stream_id {sid}"
        )

    # add downstream cell using plain python dirmap
    final_direction = int(fd_vals[path[-1][0], path[-1][1]])
    if final_direction not in (-1, -2, 0):
        drow, dcol = _ESRI_DIRMAP[final_direction]
        next_row = path[-1][0] + drow
        next_col = path[-1][1] + dcol
        if 0 <= next_row < shape[0] and 0 <= next_col < shape[1]:
            path.append((next_row, next_col))

    path_rows, path_cols = zip(*path)
    xs, ys = xy(transform, path_rows, path_cols, offset="center")
    return LineString(zip(xs, ys))
