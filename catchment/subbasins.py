import numpy as np
import xarray as xr

from .dirmap import _make_numba_esri_dirmap
from .flow_graph import propagate_downstream


def _reach_seed(stream_raster: xr.DataArray):
    """Which pixels are stream/reach seeds, and their reach id as float64
    (for propagate_downstream, which works in float64 regardless of the
    label raster's own dtype)."""
    label_values = stream_raster.values
    if np.issubdtype(label_values.dtype, np.floating):
        is_seed = np.isfinite(label_values) & (label_values != 0)
    else:
        is_seed = label_values != 0
    reach_id = np.where(is_seed, label_values, 0).astype(np.float64)
    return is_seed, reach_id


def delineate_subbasins(
    stream_raster: xr.DataArray,
    flow_directions: xr.DataArray,
) -> xr.DataArray:
    """
    Delineate subbasins: assign every pixel the id of the nearest downstream
    stream/reach it drains into.

    Multi-source flood fill over the reversed flow-direction graph, seeded
    from every stream pixel (each already carrying its own reach id) and
    expanding upstream one ring at a time. Flow direction gives every cell
    exactly one outgoing edge and is guaranteed acyclic (see
    accumulation.py), so this graph is a proper forest: each pixel has
    exactly one path to exactly one stream cell, never two candidate paths
    to compare - a plain FIFO queue suffices, no priority queue/relaxation
    needed. O(pixels) total, no pour points and no per-reach catchment
    delineation required.

    Args:
        stream_raster: Raster of channel network with unique IDs for each
            segment/reach (0 for non-stream pixels).
        flow_directions: Flow directions raster (ESRI d8 encoding)
    Returns:
        Raster of subbasins with same IDs as stream_raster
    """
    is_seed, reach_id = _reach_seed(stream_raster)
    dirmap = _make_numba_esri_dirmap()
    [reach_id_result], reached = propagate_downstream(
        is_seed, [reach_id], flow_directions, dirmap
    )

    subbasins_arr = np.where(reached, reach_id_result, 0).astype(np.int32)
    subbasins = stream_raster.copy(data=subbasins_arr)
    subbasins.encoding = {}
    return subbasins


def delineate_subbasins_and_hand(
    stream_raster: xr.DataArray,
    flow_directions: xr.DataArray,
    dem: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Delineate subbasins and compute HAND together in a single traversal.

    Both are "propagate a per-reach value to everything that drains to that
    reach": subbasins propagates the reach id, HAND propagates the stream
    cell's own elevation (then HAND = pixel elevation - propagated
    elevation). Since they share the identical traversal, this computes both
    in one flood fill rather than two.

    Returns:
        (subbasins, hand)
    """
    is_seed, reach_id = _reach_seed(stream_raster)
    elevation = dem.values.astype(np.float64)

    dirmap = _make_numba_esri_dirmap()
    (reach_id_result, elev_result), reached = propagate_downstream(
        is_seed, [reach_id, elevation], flow_directions, dirmap
    )

    subbasins_arr = np.where(reached, reach_id_result, 0).astype(np.int32)
    subbasins = stream_raster.copy(data=subbasins_arr)
    subbasins.encoding = {}

    hand_arr = elevation - elev_result
    hand_arr[~reached] = np.nan
    hand_arr[np.isnan(elevation)] = np.nan
    hand = dem.copy(data=hand_arr)
    hand.encoding = {}
    hand = hand.rio.write_nodata(np.nan)

    return subbasins, hand
