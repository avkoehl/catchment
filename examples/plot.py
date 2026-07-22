"""Generate the README figure for catchment.

Reads the committed DEM + flowlines fixture (a 30 m HUC12, downloaded once with
headwaters) and runs the full extraction, then renders a 2x3 facet of the
derived layers: row 1 is the raster routing pipeline (flow direction, flow
accumulation, stream network); row 2 is what's built on top of it (reaches,
HAND, subbasins). reaches and subbasins share IDs 1:1 (each subbasin drains to
one reach), so their panels use the same label -> color mapping.

Fixture provenance:
    headwaters.fetch_huc("180101080409", nhd_layer="medium",
                         dem_resolution=30, crs="EPSG:3310"),
    with flowlines starting within 20 m of the watershed boundary dropped
    (see headwaters/examples/plot.py).

Run:
    uv run --group example python examples/plot.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import rioxarray  # noqa: F401  (registers .rio accessor)
import xarray as xr
from matplotlib.colors import LightSource, LogNorm, hsv_to_rgb
from matplotlib.patches import Patch
from scipy.ndimage import binary_dilation

from catchment import extract_catchment

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
ASSETS = HERE.parent / "assets"

# ESRI D8 direction code -> compass angle (degrees, standard math convention:
# 0 = East, counterclockwise), used to color flow direction like an aspect map.
DIRECTIONS = {
    1: ("E", 0),
    128: ("NE", 45),
    64: ("N", 90),
    32: ("NW", 135),
    16: ("W", 180),
    8: ("SW", 225),
    4: ("S", 270),
    2: ("SE", 315),
}
SINK_COLOR = (0.6, 0.6, 0.6, 1.0)


def extent(da: xr.DataArray) -> tuple[float, float, float, float]:
    return (
        float(da.x.min()),
        float(da.x.max()),
        float(da.y.min()),
        float(da.y.max()),
    )


def masked(da: xr.DataArray) -> np.ndarray:
    a = da.values.astype(float)
    return np.where(np.isfinite(a) & (a != 0), a, np.nan)


def base_style(ax) -> None:
    ax.set_axis_off()


def categorical_colors(labels, cmap_name="nipy_spectral", seed=0) -> dict:
    """Map each label to a color, spread across the colormap and shuffled so
    spatially adjacent labels don't land on similar hues."""
    labels = sorted(labels)
    n = len(labels)
    cmap = plt.get_cmap(cmap_name)
    positions = np.linspace(0.05, 0.95, n)
    order = np.random.default_rng(seed).permutation(n)
    return {lab: cmap(positions[order[i]]) for i, lab in enumerate(labels)}


def dilate_labels(data: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Thicken single-pixel-wide stream/reach lines for legibility. Dilates
    each label's mask independently so labels never bleed into each other."""
    out = np.zeros_like(data)
    for label in np.unique(data[data > 0]):
        out[binary_dilation(data == label, iterations=iterations)] = label
    return out


def render_labeled(data: np.ndarray, color_map: dict) -> np.ndarray:
    """Render an integer-labeled array as RGBA using color_map; unmapped
    values (e.g. background 0) come out fully transparent."""
    labels_sorted = np.array(sorted(color_map.keys()))
    colors_arr = np.array([color_map[label] for label in labels_sorted])
    idx = np.clip(np.searchsorted(labels_sorted, data), 0, len(labels_sorted) - 1)
    matched = labels_sorted[idx] == data
    rgba = colors_arr[idx].copy()
    rgba[..., 3] = np.where(matched, rgba[..., 3], 0.0)
    return rgba


def panel_flow_dir(ax, hs, ext, flow_dir) -> None:
    ax.imshow(hs, cmap="gray", extent=ext, origin="upper")
    color_map = {
        code: tuple(hsv_to_rgb((angle / 360, 0.85, 0.9))) + (0.9,)
        for code, (_, angle) in DIRECTIONS.items()
    }
    color_map[-2] = SINK_COLOR
    ax.imshow(
        render_labeled(flow_dir.values, color_map), extent=ext, origin="upper"
    )
    handles = [
        Patch(color=color_map[code], label=name)
        for code, (name, _) in sorted(DIRECTIONS.items(), key=lambda kv: kv[1][1])
    ]
    ax.legend(
        handles=handles, loc="lower left", fontsize=7, ncol=2, frameon=False,
        title="flow direction", title_fontsize=8,
    )
    ax.set_title("flow direction", fontsize=11)


def panel_flow_acc(ax, hs, ext, flow_acc) -> None:
    ax.imshow(hs, cmap="gray", extent=ext, origin="upper")
    acc = masked(flow_acc)
    ax.imshow(
        acc, cmap="cubehelix_r", extent=ext, origin="upper",
        norm=LogNorm(vmin=np.nanpercentile(acc[acc > 0], 75), vmax=np.nanmax(acc)),
        alpha=0.95,
    )
    ax.set_title("flow accumulation", fontsize=11)


def panel_stream_network(ax, hs, ext, stream_network) -> None:
    ax.imshow(hs, cmap="gray", extent=ext, origin="upper")
    data = dilate_labels(stream_network.values, iterations=1)
    labels = np.unique(data[data > 0])
    color_map = categorical_colors(labels, cmap_name="tab20b", seed=1)
    rgba = render_labeled(data, color_map)
    rgba[..., 3] = np.where(rgba[..., 3] > 0, 1.0, 0.0)
    ax.imshow(rgba, extent=ext, origin="upper")
    ax.set_title(f"stream network  ({len(labels)} segments)", fontsize=11)


def panel_labeled(ax, hs, ext, raster, color_map, title, dilate: bool = False) -> None:
    ax.imshow(hs, cmap="gray", extent=ext, origin="upper")
    data = dilate_labels(raster.values, iterations=1) if dilate else raster.values
    ax.imshow(render_labeled(data, color_map), extent=ext, origin="upper")
    ax.set_title(title, fontsize=11)


def panel_hand(ax, hs, ext, hand) -> None:
    ax.imshow(hs, cmap="gray", extent=ext, origin="upper")
    im = ax.imshow(masked(hand), cmap="Spectral_r", extent=ext, origin="upper", alpha=0.9)
    ax.set_title("HAND (height above nearest drainage)", fontsize=11)
    ax.figure.colorbar(im, ax=ax, shrink=0.6, pad=0.02).set_label("m", fontsize=8)


def main() -> None:
    dem = rioxarray.open_rasterio(DATA / "dem.tif").squeeze()
    flowlines = gpd.read_file(DATA / "flowlines.gpkg")

    r = extract_catchment(dem, flowlines=flowlines)

    ext = extent(r.conditioned_dem)
    z = np.where(np.isfinite(r.conditioned_dem.values), r.conditioned_dem.values, np.nan)
    ls = LightSource(azdeg=315, altdeg=45)
    hs = ls.hillshade(z, vert_exag=2, dx=30, dy=30)

    reach_labels = np.unique(r.reaches.values[r.reaches.values > 0])
    reach_colors = categorical_colors(reach_labels, cmap_name="nipy_spectral", seed=0)

    fig, axes = plt.subplots(2, 3, figsize=(15, 11))

    panel_flow_dir(axes[0, 0], hs, ext, r.flow_dir)
    panel_flow_acc(axes[0, 1], hs, ext, r.flow_acc)
    panel_stream_network(axes[0, 2], hs, ext, r.stream_network)
    panel_labeled(axes[1, 0], hs, ext, r.reaches, reach_colors,
                  f"reaches  ({len(reach_labels)})", dilate=True)
    panel_hand(axes[1, 1], hs, ext, r.hand)
    panel_labeled(axes[1, 2], hs, ext, r.subbasins, reach_colors,
                  "subbasins  (colored to match reaches)")

    for ax in axes.flat:
        base_style(ax)

    fig.suptitle("extract_catchment  ·  derived drainage layers", fontsize=13, y=0.98)
    fig.tight_layout()

    ASSETS.mkdir(exist_ok=True)
    out = ASSETS / "catchment.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
