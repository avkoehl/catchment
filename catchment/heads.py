import numpy as np
from shapely.geometry import Point
import networkx as nx
import geopandas as gpd
import rasterio
import xarray as xr

from .nodes import _find_stream_nodes


def channel_heads_from_flow_accumulation(
    flow_acc: xr.DataArray, flow_dir: xr.DataArray, threshold_area: float
) -> xr.DataArray:
    """
    Identify channel heads from flow accumulation and flow direction rasters.

    Args:
        flow_acc (xr.DataArray): Flow accumulation raster.
        flow_dir (xr.DataArray): Flow direction raster.
        threshold_area (float): Minimum contributing area (in square units) to define a channel head.
    Returns:
        xr.DataArray: Binary raster with channel heads marked as 1, non-channel heads as 0, and nodata as 255.
    """
    num_cells = int(threshold_area / (np.abs(flow_acc.rio.resolution()[0]) ** 2))
    streams = flow_acc >= num_cells
    sources, _, _ = _find_stream_nodes(streams, flow_dir)

    channel_heads = flow_acc.copy(data=np.zeros(flow_acc.shape, dtype=np.uint8))
    for row, col in sources:
        channel_heads.data[row, col] = 1
    channel_heads.data[flow_acc == flow_acc.rio.nodata] = 255

    channel_heads = channel_heads.rio.set_nodata(255)
    channel_heads.attrs["_FillValue"] = 255
    return channel_heads


def channel_heads_from_flowlines(flowline_features, template_raster):
    """
    Identify channel heads from flowline features. Assumes flowline features are directed from upstream to downstream.

    Args:
        flowline_features (GeoDataFrame): GeoDataFrame containing flowline geometries (llinestrings).
        template_raster (xr.DataArray): Template raster to define the spatial extent and resolution of the output.
    Returns:
        xr.DataArray: Binary raster with channel heads marked as 1, non-channel heads as 0, and nodata as 255.
    """
    G = nx.DiGraph()
    for flowline in flowline_features.geometry:
        start = flowline.coords[0]
        end = flowline.coords[-1]
        G.add_edge(start, end)

    channel_heads_points = [Point(node) for node, deg in G.in_degree() if deg == 0]
    channel_heads_points = gpd.GeoSeries(
        channel_heads_points, crs=flowline_features.crs
    )

    rows, cols = rasterio.transform.rowcol(
        template_raster.rio.transform(),
        channel_heads_points.x.values,
        channel_heads_points.y.values,
        op=int,
    )

    h, w = template_raster.shape
    valid = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    binary = np.zeros((h, w), dtype=np.uint8)
    binary[rows[valid], cols[valid]] = 1
    binary[template_raster == template_raster.rio.nodata] = 255

    result = template_raster.copy(data=binary)
    result = result.rio.set_nodata(255)
    result.attrs["_FillValue"] = 255

    return result
