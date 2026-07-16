import numba
import numpy as np
import xarray as xr

from .dirmap import _make_numba_esri_dirmap
from .nodes import _find_stream_nodes


def extract_channel_network(
    channel_heads: xr.DataArray, flow_dir: xr.DataArray
) -> xr.DataArray:
    """
    Extract the channel network from a set of channel head locations and flow direction data.

    Args:
        channel_heads: xarray DataArray of channel head points (binary raster).
        flow_dir: xarray DataArray of flow directions (ESRI style).
    Returns:
        xarray DataArray where each stream segment between junctions has a unique positive integer ID, with non-stream pixels as 0.
    """
    streams = trace_streams(channel_heads, flow_dir)
    labeled_streams = link_streams(streams, flow_dir)
    return labeled_streams


def link_streams(
    stream_raster: xr.DataArray, flow_directions: xr.DataArray
) -> xr.DataArray:
    """Assign unique IDs to stream segments between junctions.

    Args:
        stream_raster: Binary or labeled stream network (non-zero values are
            streams, zero values are non-stream pixels).
        flow_directions: Flow direction raster in D8 format (ESRI convention).

    Returns:
        A raster where each stream segment between junctions has a unique positive integer ID, with non-stream pixels as 0.
    """
    dirmap = _make_numba_esri_dirmap()
    sources, confluences, _ = _find_stream_nodes(stream_raster, flow_directions)
    link_arr = _link_streams_numba(
        stream_raster.data, flow_directions.data, dirmap, sources, confluences
    )
    link_raster = flow_directions.copy(data=link_arr)
    return link_raster


@numba.njit
def _link_streams_numba(stream_arr, flow_directions_arr, dirmap, sources, confluences):
    """Assign unique IDs to stream links (segments between junctions)"""
    nrows, ncols = flow_directions_arr.shape

    # Create confluence lookup for faster checking
    confluence_arr = np.zeros((nrows, ncols), dtype=np.uint8)
    for row, col in confluences:
        confluence_arr[row, col] = 1

    # Assign link IDs starting from each source
    link_id = 1
    link_arr = np.zeros((nrows, ncols), dtype=np.uint16)

    for source_row, source_col in sources:
        row, col = source_row, source_col

        while True:
            if link_arr[row, col] != 0 or stream_arr[row, col] == 0:
                break

            link_arr[row, col] = link_id

            current_direction = flow_directions_arr[row, col]
            if current_direction in (-1, -2, 0):
                link_id += 1
                break

            drow, dcol = dirmap[current_direction]
            next_row = row + drow
            next_col = col + dcol

            if not (0 <= next_row < nrows and 0 <= next_col < ncols):
                link_id += 1
                break

            # If next cell is a confluence, end current link
            if confluence_arr[next_row, next_col] == 1:
                link_id += 1
                row, col = next_row, next_col
                continue

            # If next cell already has an ID, we've merged
            if link_arr[next_row, next_col] != 0:
                link_id += 1
                break

            row, col = next_row, next_col

    return link_arr


def trace_streams(
    channel_heads: xr.DataArray, flow_directions: xr.DataArray
) -> xr.DataArray:
    """
    Trace streams from a set of channel head locations using flow direction data.
    Args:
        channel_heads: xarray DataArray of channel head points (binary raster).
        flow_directions: xarray DataArray of flow directions (ESRI style).
    Returns:
        Binary stream raster where 1 indicates stream cells and 0 indicates non-stream cells. Can be used as input to `link_streams` to label individual stream segments.
    """
    points = np.column_stack(np.where(channel_heads.data == 1))
    dirmap = _make_numba_esri_dirmap()
    stream_arr = _trace_streams_numba(points, flow_directions.data, dirmap)
    stream_raster = flow_directions.copy(data=stream_arr)
    return stream_raster


@numba.njit
def _trace_streams_numba(points, flow_directions_arr, dirmap):
    """Mark all stream cells (binary stream network)"""
    nrows, ncols = flow_directions_arr.shape
    stream_arr = np.zeros((nrows, ncols), dtype=np.uint8)

    for point in points:
        row, col = point
        if stream_arr[row, col] != 0:
            continue

        while True:
            stream_arr[row, col] = 1

            current_direction = flow_directions_arr[row, col]
            if current_direction in (-1, -2, 0):
                break

            drow, dcol = dirmap[current_direction]
            next_row = row + drow
            next_col = col + dcol

            if not (0 <= next_row < nrows and 0 <= next_col < ncols):
                break

            if stream_arr[next_row, next_col] != 0:
                break

            row, col = next_row, next_col

    return stream_arr
