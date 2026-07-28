from typing import NamedTuple

import geopandas as gpd
import xarray as xr

from .accumulation import flow_accumulation_workflow
from .heads import channel_heads_from_flow_accumulation, channel_heads_from_flowlines
from .network import extract_channel_network
from .subbasins import delineate_subbasins_and_hand
from .reaches import delineate_reaches
from .stream_to_vector import streams_to_vector


class CatchmentResults(NamedTuple):
    conditioned_dem: xr.DataArray
    flow_dir: xr.DataArray
    flow_acc: xr.DataArray
    channel_heads: xr.DataArray
    stream_network: xr.DataArray
    subbasins: xr.DataArray
    reaches: xr.DataArray
    hand: xr.DataArray
    vector_network: gpd.GeoDataFrame


def extract_catchment(
    dem: xr.DataArray,
    flowlines: gpd.GeoDataFrame | None = None,
    threshold_area: float | None = None,
    reach_kwargs: dict | None = None,
) -> CatchmentResults:
    """Given a raw DEM, extract the full raster description of the drainage
    system: conditioned DEM, flow direction/accumulation, labeled channel
    network, subbasins, reaches, HAND, and the vectorized stream network.

    Channel initiation points come from one of two sources: the upstream
    endpoints of `flowlines`, or a contributing-area threshold applied to the
    computed flow accumulation. Exactly one of `flowlines` or `threshold_area`
    must be provided.

    Args:
        dem: Digital elevation model (raw; conditioning is done internally).
        flowlines: Flowline features directed upstream to downstream, used to
            derive channel initiation points.
        threshold_area: Minimum contributing area (in square map units) to
            define a channel head.
        reach_kwargs: Optional keyword arguments passed to delineate_reaches.

    Returns:
        CatchmentResults(conditioned_dem, flow_dir, flow_acc, channel_heads,
        stream_network, subbasins, reaches, hand, vector_network)
    """
    if (flowlines is None) == (threshold_area is None):
        raise ValueError("provide exactly one of flowlines or threshold_area")

    conditioned_dem, flow_dir, flow_acc = flow_accumulation_workflow(dem)

    if flowlines is not None:
        channel_heads = channel_heads_from_flowlines(flowlines, flow_dir)
    else:
        channel_heads = channel_heads_from_flow_accumulation(
            flow_acc, flow_dir, threshold_area
        )

    stream_network = extract_channel_network(channel_heads, flow_dir)
    reaches = delineate_reaches(
        stream_network, conditioned_dem, flow_dir, flow_acc, **(reach_kwargs or {})
    )

    subbasins, hand = delineate_subbasins_and_hand(reaches, flow_dir, conditioned_dem)
    vector_network = streams_to_vector(stream_network, flow_dir, flow_acc)

    return CatchmentResults(
        conditioned_dem=conditioned_dem,
        flow_dir=flow_dir,
        flow_acc=flow_acc,
        channel_heads=channel_heads,
        stream_network=stream_network,
        subbasins=subbasins,
        reaches=reaches,
        hand=hand,
        vector_network=vector_network,
    )
