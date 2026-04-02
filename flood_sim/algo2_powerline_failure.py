'''
Algorithm 2 (Flood): Power Line Failure Assessment

    Input:
        - Flood data F_df: synthetic_data/flood_data.geojson
        - Power lines G_lines: synthetic_data/powerline.geojson
        - Fragility parameters (μ, σ): lognormal, calibrated to flood depth
        - Time horizon T: 24 hours
    Output:
        - L_status : np.ndarray (T, N)  1 = intact, 0 = failed
        - L_depth  : np.ndarray (T, N)  local flood depth at each line midpoint
'''

import json
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.collections as mc
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Circle
from math import log, erfc, sqrt


# Fragility function for flood situation: same as windstorm version but using flood depth instead of wind speed.
# x is now flood depth (m) instead of wind speed (m/s)
def lognormal_cdf_powerline(x, mu, sigma):
    """
    Lognormal CDF fragility — eq. (4) in the paper.

    Parameters
    ----------
    x     : local flood depth at the line (metres)
    mu    : ln-mean  (calibrated to flood depth)
    sigma : ln-std   (calibrated to flood depth)

    Returns
    -------
    p_fail : probability of failure at depth x
    """
    if x <= 0:
        return 0.0
    z = (log(x) - mu) / (sigma * sqrt(2))
    return 0.5 * erfc(-z)


def load_data(
    flood_path      = "synthetic_data/flood_data.geojson",
    powerline_path  = "synthetic_data/powerline.geojson",
    substation_path = "synthetic_data/substation.geojson",
    boundary_path   = "jsons/Precinct_Boundaries.geojson",
    epsg            = 32618,
):
    """
    Load all spatial data and reproject to a common CRS.

    Returns
    -------
    flood_data   : list of dicts  {t, origin (x,y), flood_radius, max_depth}
    lines_proj   : GeoDataFrame   power lines in projected CRS
    subs_proj    : GeoDataFrame   substations in projected CRS
    boundary_gdf : GeoDataFrame   study boundary in projected CRS
    """
    with open(flood_path) as f:
        raw = json.load(f)

    flood_data = [
        {
            "t":            feat["properties"]["t"],
            "origin":       tuple(feat["geometry"]["coordinates"]),
            "flood_radius": feat["properties"]["flood_radius"],
            "max_depth":    feat["properties"]["max_depth"],
        }
        for feat in raw["features"]
    ]

    lines_proj   = gpd.read_file(powerline_path).to_crs(epsg=epsg)
    subs_proj    = gpd.read_file(substation_path).to_crs(epsg=epsg)
    boundary_gdf = gpd.read_file(boundary_path).to_crs(epsg=epsg)

    return flood_data, lines_proj, subs_proj, boundary_gdf


def assess_powerline_failures(
    flood_data,
    lines_gdf,
    mu    = -0.22,  # ln-mean: 50% failure probability at ~0.8m depth
    sigma = 0.30,   # ln-std
    seed  = 42,
):
    """
    Algorithm 2 (Flood): Power Line Failure Assessment.

    Key difference from windstorm version:
        - The flooded zone is PERSISTENT — once a line is inundated it stays
          inundated for all subsequent timesteps (flood_radius never shrinks).
        - Every already-flooded intact line is re-evaluated each hour because
          continued submersion keeps applying stress, not just the moment of
          first inundation.

    Parameters
    ----------
    flood_data : list of dicts  {t, origin, flood_radius, max_depth}
    lines_gdf  : GeoDataFrame   must have columns mid_x, mid_y
    mu, sigma  : lognormal fragility parameters (depth-based)
    seed       : random seed for reproducibility

    Returns
    -------
    L_status : np.ndarray (T, N)  1 = intact, 0 = failed
    L_depth  : np.ndarray (T, N)  local flood depth at each line midpoint
    """

    T       = len(flood_data)
    n_lines = len(lines_gdf)
    mid_x   = lines_gdf["mid_x"].values.astype(float)
    mid_y   = lines_gdf["mid_y"].values.astype(float)

    L_status      = np.ones((T, n_lines), dtype=int)
    L_depth       = np.zeros((T, n_lines))
    flooded_mask  = np.zeros(n_lines, dtype=bool)   # persistent: once True, stays True
    rng           = np.random.default_rng(seed)

    for t, flood in enumerate(flood_data):
        ox, oy        = flood["origin"]
        flood_radius  = flood["flood_radius"]
        max_depth     = flood["max_depth"]

        # Carry forward previous failure state
        if t > 0:
            L_status[t] = L_status[t - 1]

        # Distance from each line midpoint to flood origin
        dist = np.sqrt((mid_x - ox) ** 2 + (mid_y - oy) ** 2)

        # Newly inundated lines this timestep
        newly_flooded = (dist < flood_radius) & (~flooded_mask)
        flooded_mask |= newly_flooded           # permanently mark as flooded

        # All currently flooded lines get a depth value
        # Depth: max at origin, linearly decays to 0 at the flood edge
        flooded_indices = np.where(flooded_mask)[0]
        for i in flooded_indices:
            depth        = max_depth * max(0.0, 1.0 - dist[i] / flood_radius)
            L_depth[t, i] = depth

            # Re-evaluate every flooded intact line each timestep —
            # continued submersion keeps stressing the line
            if L_status[t, i] == 1:
                p_fail = lognormal_cdf_powerline(depth, mu, sigma)
                if rng.random() < p_fail:
                    L_status[t:, i] = 0     # failed — remains offline

    return L_status, L_depth


def animate_failures(
    flood_data,
    lines_proj,
    L_status,
    subs_proj    = None,
    boundary_gdf = None,
    n_frames     = 120,
    interval     = 150,
    repeat       = False,
):
    """
    Animate the power line failure assessment over time.

    Animation frames are interpolated across the 24-hour flood_data so the
    flood circle expands smoothly (same approach as flood_generate.py).

    Parameters
    ----------
    flood_data   : list of dicts (output of load_data)
    lines_proj   : GeoDataFrame  projected power lines
    L_status     : np.ndarray    (T, N) output of assess_powerline_failures
    subs_proj    : GeoDataFrame  (optional) substations
    boundary_gdf : GeoDataFrame  (optional) study boundary
    n_frames     : int           total animation frames
    interval     : int           milliseconds between frames
    repeat       : bool          whether to loop
    """

    T       = len(flood_data)
    n_lines = len(lines_proj)

    # Interpolate flood radius and depth across n_frames
    data_t   = np.array([s["t"]            for s in flood_data], dtype=float)
    radii    = np.array([s["flood_radius"]  for s in flood_data], dtype=float)
    depths   = np.array([s["max_depth"]     for s in flood_data], dtype=float)
    hours    = np.linspace(data_t[0], data_t[-1], n_frames)
    r_interp = np.interp(hours, data_t, radii)
    d_interp = np.interp(hours, data_t, depths)

    # Map each animation frame to the nearest data timestep for L_status
    t_indices = np.round(np.linspace(0, T - 1, n_frames)).astype(int)

    origin     = flood_data[0]["origin"]
    line_segs  = [list(geom.coords) for geom in lines_proj.geometry]

    fig, ax = plt.subplots(figsize=(11, 9))

    if boundary_gdf is not None:
        boundary_gdf.boundary.plot(ax=ax, linewidth=1, color="black", zorder=1)
    if subs_proj is not None:
        subs_proj.plot(ax=ax, color="blue", markersize=20, marker="s", zorder=4)

    lc = mc.LineCollection(line_segs, colors="limegreen", linewidths=1.5, zorder=2)
    ax.add_collection(lc)

    flood_fill = Circle(origin, 0, color="steelblue", alpha=0.20, zorder=3)
    flood_ring = Circle(origin, 0, fill=False, edgecolor="steelblue", linewidth=1.5, zorder=3)
    ax.add_patch(flood_fill)
    ax.add_patch(flood_ring)
    ax.plot(*origin, "r^", markersize=10, zorder=5, label="Flood origin")

    net_bounds = lines_proj.total_bounds
    margin     = 8000
    ax.set_xlim(net_bounds[0] - margin, net_bounds[2] + margin)
    ax.set_ylim(net_bounds[1] - margin, net_bounds[3] + margin)
    ax.set_aspect("equal")
    ax.set_title("Algorithm 2 (Flood) — Power Line Failure Assessment", fontsize=13)
    ax.set_xlabel("X  (m, EPSG:32618)")
    ax.set_ylabel("Y  (m, EPSG:32618)")

    info = ax.text(
        0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8), zorder=6,
    )

    legend_handles = [
        Line2D([0], [0], color="limegreen", lw=2, label="Intact line"),
        Line2D([0], [0], color="red",       lw=2, label="Failed line"),
        Patch(facecolor="steelblue", alpha=0.2, edgecolor="steelblue", label="Flood extent"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="blue",
               markersize=8, label="Substation"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8)

    def update(frame):
        t_idx    = t_indices[frame]
        colors   = np.where(L_status[t_idx] == 1, "limegreen", "red")
        lc.set_colors(colors)

        r = r_interp[frame]
        flood_fill.set_radius(r)
        flood_ring.set_radius(r)

        n_failed = int(np.sum(L_status[t_idx] == 0))
        info.set_text(
            f"Hour        : {hours[frame]:>5.1f}\n"
            f"Flood radius: {r/1000:>6.2f} km\n"
            f"Depth origin: {d_interp[frame]:>5.2f} m\n"
            f"Failed      : {n_failed:>3d} / {n_lines}"
        )
        return lc, flood_fill, flood_ring, info

    anim = FuncAnimation(fig, update, frames=n_frames,
                         interval=interval, blit=False, repeat=repeat)
    plt.tight_layout()
    return anim


if __name__ == "__main__":
    flood_data, lines_proj, subs_proj, boundary_gdf = load_data()

    L_status, L_depth = assess_powerline_failures(flood_data, lines_proj)

    anim = animate_failures(
        flood_data, lines_proj, L_status,
        subs_proj    = subs_proj,
        boundary_gdf = boundary_gdf,
    )
    plt.show()
