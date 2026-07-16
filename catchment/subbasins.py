import numpy as np
import pandas as pd
from scipy.ndimage import maximum_position, maximum as ndimage_maximum
import xarray as xr

from .adapters import to_pysheds


def delineate_subbasins(
    stream_raster: xr.DataArray,
    flow_directions: xr.DataArray,
    flow_accumulation: xr.DataArray,
) -> xr.DataArray:
    """
    Delineate all subbasins given a channel network raster. Detects pour points
    for each unique stream segment by finding the cell with the highest flow
    accumulation for that segment (based on ID).

    Args:
        stream_raster: Raster of channel network with unique IDs for each
            segment
        flow_directions: Flow directions raster (ESRI d8 encoding)
        flow_accumulation: Flow accumulation raster
    Returns:
        Raster of subbasins with same IDs as stream_raster
    """
    # get pour points from channel network raster
    # these are sorted so that nested basins are handled correctly
    pour_points = _identify_pour_points(stream_raster, flow_accumulation)

    subbasins = stream_raster.copy(
        data=np.zeros_like(stream_raster.data, dtype=np.int32)
    )

    pysheds_fdir, grid = to_pysheds(flow_directions)

    for _, row in pour_points.iterrows():
        pour_row = row["row"]
        pour_col = row["col"]

        catchment = grid.catchment(
            x=pour_col,
            y=pour_row,
            fdir=pysheds_fdir,
            xytype="index",
        )

        subbasins.data[catchment] = row["stream_value"]

    return subbasins


def _identify_pour_points(stream_raster, flow_accumulation):
    labels = stream_raster.data.astype(np.int32)
    fa = flow_accumulation.data

    unique_ids = np.unique(labels)
    unique_ids = unique_ids[(unique_ids > 0) & ~np.isnan(unique_ids.astype(float))]

    max_fa = ndimage_maximum(fa, labels=labels, index=unique_ids)
    positions = maximum_position(fa, labels=labels, index=unique_ids)

    pour_points = pd.DataFrame(
        {
            "row": [int(p[0]) for p in positions],
            "col": [int(p[1]) for p in positions],
            "flow_accumulation": [float(v) for v in max_fa],
            "stream_value": unique_ids.astype(int),
        }
    )
    pour_points = pour_points.sort_values("flow_accumulation", ascending=False)
    return pour_points
