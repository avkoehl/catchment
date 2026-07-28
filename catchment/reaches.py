import numpy as np
import pandas as pd
from rasterio.transform import xy
import ruptures as rpt
import xarray as xr

from .routing import route_stream
from .raster_index import group_pixels_by_id


def delineate_reaches(
    stream_raster: xr.DataArray,
    dem: xr.DataArray,
    flow_dir: xr.DataArray,
    flow_acc: xr.DataArray,
    penalty: float | None = None,
    min_length: float = 500,
    smooth_window: int | None = None,
    threshold_degrees: float = 1.0,
) -> xr.DataArray:
    """Segment stream networks into reaches based on slope change points.

    Uses the PELT (Pruned Exact Linear Time) changepoint detection algorithm
    to identify distinct reaches along each stream based on slope variations.
    Adjacent reaches with similar slopes can be merged using the threshold parameter.

    Args:
        stream_raster: Labeled stream network where each unique value represents
            a stream segment (0 for non-stream pixels).
        dem: Digital elevation model for slope calculations.
        flow_dir: Flow direction raster corresponding to the DEM.
        flow_acc: Flow accumulation raster corresponding to the DEM.
        penalty: PELT algorithm penalty parameter. If None, automatically calculated
            as log(n) * variance of slope signal.
        min_length: Minimum reach length in meters.
        smooth_window: Window size for smoothing slope values before segmentation.
            If None, no smoothing is applied.
        threshold_degrees: Merge adjacent reaches if slope difference is below this
            threshold in degrees.

    Returns:
        A raster where each pixel value represents a unique reach ID (0 for non-stream pixels). Reach IDs are computed as reach_number + (stream_id * 1000).
    """
    reaches = stream_raster.copy(data=np.zeros_like(stream_raster, dtype=np.uint32))
    label_raster = stream_raster.values
    pixel_index = group_pixels_by_id(label_raster)
    for stream_val, (rows, cols) in pixel_index.items():
        if len(rows) == 1:  # single pixel stream
            reach_id = np.uint32(stream_val) * 1000
            reaches.data[rows, cols] = reach_id
            continue

        stream_df = _create_stream_points(
            rows, cols, label_raster, stream_val, flow_dir, flow_acc, dem
        )
        # roughly convert min_length in meters to number of points
        min_size = int(min_length / flow_dir.rio.resolution()[0])
        stream_df = _pelt_reaches(
            stream_df,
            penalty=penalty,
            min_size=min_size,
            smooth_window=smooth_window,
        )
        stream_df = _merge_reaches_by_threshold(
            stream_df, threshold_degrees=threshold_degrees
        )
        stream_df["reach_val"] = stream_df["reach_id"] + (
            np.uint32(stream_val) * 1000
        )
        out_rows, out_cols = stream_df["row"].values, stream_df["col"].values
        reaches.data[out_rows, out_cols] = stream_df["reach_val"].values
    return reaches


def _pelt_reaches(stream_df, penalty, min_size, smooth_window, model="rbf"):
    if len(stream_df) < min_size:
        stream_df["reach_id"] = 0
        return stream_df

    if smooth_window:
        slopes = (
            stream_df["slope_degrees"]
            .rolling(smooth_window, center=True, min_periods=1)
            .mean()
            .values
        )
    else:
        slopes = stream_df["slope_degrees"].values

    signal = slopes.reshape(-1, 1)
    algo = rpt.Pelt(model=model, min_size=min_size)

    if penalty is None:
        penalty = np.log(len(signal)) * signal.var()

    cp = algo.fit(signal).predict(pen=penalty)
    cp = cp[:-1]  # remove last point which is length of signal

    if len(cp) == 0:
        stream_df["reach_id"] = 0
    else:
        stream_df["reach_id"] = np.searchsorted(cp, np.arange(len(stream_df)))
    return stream_df


def _merge_reaches_by_threshold(stream_df, threshold_degrees=1.0):
    """
    Merge adjacent reaches if slope change is less than threshold.

    Parameters:
    -----------
    stream_df : pd.DataFrame
        DataFrame with reach_id and slope_degrees columns
    threshold_degrees : float
        Minimum slope change (in degrees) to keep a reach boundary

    Returns:
    --------
    stream_df : pd.DataFrame
        DataFrame with updated reach_id column
    """
    # Calculate median slope for each reach
    reach_stats = stream_df.groupby("reach_id")["slope_degrees"].median()

    # Keep merging until all boundaries meet threshold
    while True:
        merged = False

        # Check each boundary between adjacent reaches
        for reach_id in sorted(reach_stats.index[:-1]):
            next_reach_id = reach_id + 1

            if next_reach_id not in reach_stats.index:
                continue

            slope_diff = abs(reach_stats[reach_id] - reach_stats[next_reach_id])

            # If difference is below threshold, merge
            if slope_diff < threshold_degrees:
                # Merge next_reach_id into reach_id
                stream_df.loc[stream_df["reach_id"] == next_reach_id, "reach_id"] = (
                    reach_id
                )

                # Recalculate median slope for merged reach
                reach_stats[reach_id] = stream_df[stream_df["reach_id"] == reach_id][
                    "slope_degrees"
                ].median()
                reach_stats = reach_stats.drop(next_reach_id)

                merged = True
                break  # Restart the loop after a merge

        if not merged:
            break  # No more merges possible

    # Renumber reach_ids to be sequential from 0
    unique_reaches = sorted(stream_df["reach_id"].unique())
    reach_map = {old_id: new_id for new_id, old_id in enumerate(unique_reaches)}
    stream_df["reach_id"] = stream_df["reach_id"].map(reach_map)

    return stream_df


def _create_stream_points(rows, cols, label_raster, target_id, flow_dir, flow_acc, dem):
    def calculate_gradient(elevations, distances):
        gradient = np.gradient(elevations, distances)
        slope_degrees = np.degrees(np.arctan(gradient))
        return np.abs(slope_degrees)

    def calculate_distance_along_stream(xs, ys):
        dx = np.diff(xs, prepend=xs[0])
        dy = np.diff(ys, prepend=ys[0])
        distances = np.sqrt(dx**2 + dy**2)
        return np.cumsum(distances)

    path = route_stream(rows, cols, label_raster, target_id, flow_dir, flow_acc)
    rows, cols = zip(*path)
    rows = np.array(rows)
    cols = np.array(cols)

    xs, ys = xy(flow_acc.rio.transform(), rows, cols, offset="center")
    stream_df = pd.DataFrame({"x": xs, "y": ys, "row": rows, "col": cols})
    stream_df["point_id"] = range(len(stream_df))
    stream_df["distance"] = calculate_distance_along_stream(xs, ys)
    stream_df["elevation"] = dem.values[rows, cols]
    stream_df["slope_degrees"] = calculate_gradient(
        stream_df["elevation"].values, stream_df["distance"].values
    )
    return stream_df
