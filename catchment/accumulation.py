import tempfile
import os
import time
import shutil

import numpy as np
import rioxarray as rxr
import xarray as xr
import whitebox

from .adapters import to_pysheds, from_pysheds


def condition_dem(dem, max_retries=3, wait_time=1.0):
    """Condition DEM using WhiteboxTools fill_depressions with retry logic."""
    for attempt in range(1, max_retries + 1):
        try:
            # Create fresh temp dir and WBT instance
            working_dir = tempfile.mkdtemp()
            wbt = whitebox.WhiteboxTools()
            wbt.set_working_dir(working_dir)
            wbt.verbose = False

            # Write DEM to disk
            dem_path = os.path.join(working_dir, "dem.tif")
            filled_path = os.path.join(working_dir, "filled_dem.tif")
            dem.rio.to_raster(dem_path)

            # Run fill depressions
            wbt.fill_depressions(dem_path, filled_path, fix_flats=True)

            # Check for output
            if not os.path.exists(filled_path) or os.path.getsize(filled_path) == 0:
                raise FileNotFoundError("WhiteboxTools failed to produce output")

            # Load result
            conditioned_dem = rxr.open_rasterio(filled_path, masked=True).squeeze().load()
            shutil.rmtree(working_dir, ignore_errors=True)
            return conditioned_dem

        except Exception as e:
            print(f"[Attempt {attempt}/{max_retries}] fill_depressions failed: {e}")
            if attempt < max_retries:
                time.sleep(wait_time)
                continue
            else:
                raise RuntimeError(
                    f"condition_dem failed after {max_retries} attempts"
                ) from e


def flow_accumulation_workflow(
    dem: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Given a DEM, compute the conditioned DEM, flow directions, and flow
    accumulation. Uses d8 flow directions, wraps around whiteboxtools 'fill
    depression with fix flats' algorithm. Flow direction and accumulation done
    with pysheds. Uses ESRI flow direction encoding.

    Args:
        dem: DEM raster
    Returns:
        (conditioned DEM, flow directions, and flow accumulation)

    """
    # wbt condition
    conditioned_dem = condition_dem(dem)
    pysheds_conditioned_dem, grid = to_pysheds(conditioned_dem)
    flow_directions = grid.flowdir(pysheds_conditioned_dem)
    flow_accumulation = grid.accumulation(flow_directions)

    cdem = from_pysheds(pysheds_conditioned_dem)
    flow_directions = from_pysheds(flow_directions)
    flow_directions = flow_directions.astype(np.int16)
    flow_accumulation = from_pysheds(flow_accumulation)
    return (
        cdem.rio.reproject_match(dem),
        flow_directions.rio.reproject_match(dem),
        flow_accumulation.rio.reproject_match(dem),
    )
