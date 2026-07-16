"""Adapter functions to convert between rioxarray DataArray and pysheds Grid."""

import rioxarray as rxr
import xarray as xr
from pysheds.grid import Grid
from pysheds.view import Raster, ViewFinder


def to_pysheds(raster_xr):
    """Convert rioxarray DataArray to pysheds Grid."""
    affine = raster_xr.rio.transform()
    crs = raster_xr.rio.crs
    nodata = raster_xr.rio.nodata

    viewfinder = ViewFinder(
        affine=affine, shape=raster_xr.shape, crs=crs, nodata=nodata
    )
    raster = Raster(raster_xr.data, viewfinder=viewfinder)
    grid = Grid(viewfinder=viewfinder)
    return raster, grid


def from_pysheds(pysheds_raster):
    viewfinder = pysheds_raster.viewfinder

    # Create coordinate arrays from the viewfinder's affine transform
    affine = viewfinder.affine
    height, width = viewfinder.shape

    # Calculate x and y coordinates
    x_coords = [affine.c + affine.a * (i + 0.5) for i in range(width)]
    y_coords = [affine.f + affine.e * (j + 0.5) for j in range(height)]

    # Create the DataArray
    raster_xr = xr.DataArray(
        pysheds_raster.data, dims=["y", "x"], coords={"y": y_coords, "x": x_coords}
    )

    # Set spatial reference information
    raster_xr = raster_xr.rio.write_crs(viewfinder.crs)
    raster_xr = raster_xr.rio.write_transform(affine)

    if viewfinder.nodata is not None:
        raster_xr = raster_xr.rio.write_nodata(viewfinder.nodata)

    return raster_xr
