import geopandas as gpd
import xarray as xr

from .network import extract_channel_network
from .subbasins import delineate_subbasins
from .reaches import delineate_reaches
from .stream_to_vector import streams_to_vector


def extract_catchment(
    dem: xr.DataArray,
    channel_heads: xr.DataArray,
    flow_dir: xr.DataArray,
    flow_acc: xr.DataArray,
    reach_kwargs: dict | None = None,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, gpd.GeoDataFrame]:
    """Given a DEM, channel heads, flow directions, and flow accumulation,
    extract the labeled channel network, subbasins, reaches, and vectorized
    stream network.

    Args:
        dem: Digital elevation model.
        channel_heads: Binary raster of channel head locations.
        flow_dir: Flow direction raster (ESRI d8 encoding).
        flow_acc: Flow accumulation raster.
        reach_kwargs: Optional keyword arguments passed to delineate_reaches.

    Returns:
        (stream_network, subbasins, reaches, vector_network)
    """
    stream_network = extract_channel_network(channel_heads, flow_dir)
    subbasins = delineate_subbasins(stream_network, flow_dir, flow_acc)
    reaches = delineate_reaches(
        stream_network, dem, flow_dir, flow_acc, **(reach_kwargs or {})
    )
    vector_network = streams_to_vector(stream_network, flow_dir, flow_acc)
    return stream_network, subbasins, reaches, vector_network
