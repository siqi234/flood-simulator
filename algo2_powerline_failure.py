'''
Algorithm 2: Power Line Failure Assessment

    Input: 
        - Storm data S_df: use a synthetic data
        - Power lines G_lines: use a synthetic data
        - Fragility parameters (μ, σ): use synthetic parameters
        - Time horizon T: 24 hours
    Output:
        - Line status L_status: {line_id: status (0 for intact, 1 for failed)}
        - Local wind speeds L_wind: {line_id: wind speed at line location}

 '''
import json
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.collections as mc
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from math import log, erfc, sqrt


# Define the fragility function for power lines from eq. (4) in the paper
def lognormal_cdf_powerline(x, mu, sigma):
    """
        Lognormal CDF — eq. (4) in the paper.

        P(X <= x) for X ~ LogNormal(mu, sigma)
        where mu is the ln-mean and sigma is the ln-std.

        Parameters:
            - x     : local wind speed at the line
            - mu    : ln-mean of the fragility function
            - sigma : ln-std of the fragility function
        Returns:
            - p_fail : probability of failure for the line at wind speed x

    """

    if x <= 0:
        return 0.0
    
    z = (log(x) - mu) / (sigma * sqrt(2))
    return 0.5 * erfc(-z)

# Data loading and preprocessing
def load_data(
    storm_path      = "synthetic_data/storm_data.geojson",
    powerline_path  = "synthetic_data/powerline.geojson",
    substation_path = "synthetic_data/substation.geojson",
    boundary_path   = "jsons/Precinct_Boundaries.geojson",
    epsg            = 32618,
):
    """
        Load all spatial data and reproject to a common CRS.

        Returns
        -------
        storm_data   : list of dicts  {t, center (x,y), radius, wind_speed}
        lines_proj   : GeoDataFrame   power lines in projected CRS
        subs_proj    : GeoDataFrame   substations in projected CRS
        boundary_gdf : GeoDataFrame   study boundary in projected CRS
    """
    with open(storm_path) as f:
        raw = json.load(f)

    storm_data = [
        {
            "t":          feat["properties"]["t"],
            "center":     tuple(feat["geometry"]["coordinates"]),
            "radius":     feat["properties"]["radius"],
            "wind_speed": feat["properties"]["wind_speed"],
        }
        for feat in raw["features"]
    ]

    lines_proj   = gpd.read_file(powerline_path).to_crs(epsg=epsg)
    subs_proj    = gpd.read_file(substation_path).to_crs(epsg=epsg)
    boundary_gdf = gpd.read_file(boundary_path).to_crs(epsg=epsg)

    return storm_data, lines_proj, subs_proj, boundary_gdf


# Algorithm 2: Power Line Failure Assessment
def assess_powerline_failures(
    storm_data,
    lines_gdf,
    mu    = 3.81,   # ln-mean  (Table II: Line Fragility Mean)
    sigma = 0.18,   # ln-std   (Table II: Line Fragility Std)
    seed  = 42,
):
    """
        Algorithm 2: Power Line Failure Assessment.

        Parameters:
            - storm_data : list of dicts  {t, center, radius, wind_speed}
            - lines_gdf  : GeoDataFrame   must have columns mid_x, mid_y (projected coords)
            - mu, sigma  : lognormal fragility parameters (eq. 4)
            - seed       : random seed for reproducibility

        Returns:
            - L_status : np.ndarray shape (T, N)  1 = intact, 0 = failed
            - L_wind   : np.ndarray shape (T, N)  local wind speed at each line midpoint

    """

    T       = len(storm_data)
    n_lines = len(lines_gdf)
    mid_x   = lines_gdf["mid_x"].values.astype(float)
    mid_y   = lines_gdf["mid_y"].values.astype(float)

    L_status = np.ones((T, n_lines), dtype=int)
    L_wind   = np.zeros((T, n_lines))
    rng      = np.random.default_rng(seed)

    for t, storm in enumerate(storm_data):
        cx, cy = storm["center"]
        r      = storm["radius"]
        w_gust = storm["wind_speed"]

        dist   = np.sqrt((mid_x - cx) ** 2 + (mid_y - cy) ** 2)
        inside = np.where(dist < r)[0]

        for i in inside:
            # Distance-based wind decay: 100% at center, 50% at edge
            decay        = 1.0 - 0.5 * dist[i] / r
            w_local      = w_gust * decay
            L_wind[t, i] = w_local

            if L_status[t, i] == 1:                          # only evaluate intact lines
                p_fail = lognormal_cdf_powerline(w_local, mu, sigma)
                if rng.random() < p_fail:
                    L_status[t:, i] = 0                      # failed — remains offline

    return L_status, L_wind


# Visualization and animation
def animate_failures(
    storm_data,
    lines_proj,
    L_status,
    subs_proj    = None,
    boundary_gdf = None,
    interval     = 600,
    repeat       = False,
):
    """
        Animate the power line failure assessment over time.

        Parameters:
            - storm_data   : list of dicts (output of load_data)
            - lines_proj   : GeoDataFrame  projected power lines
            - L_status     : np.ndarray    (T, N) output of assess_powerline_failures
            - subs_proj    : GeoDataFrame  (optional) substations to overlay
            - boundary_gdf : GeoDataFrame  (optional) study boundary to overlay
            - interval     : int           milliseconds between frames
            - repeat       : bool          whether to loop the animation

        Returns:
            - anim : matplotlib FuncAnimation

    """

    T       = len(storm_data)
    n_lines = len(lines_proj)
    line_segs = [list(geom.coords) for geom in lines_proj.geometry]

    fig, ax = plt.subplots(figsize=(11, 9))

    if boundary_gdf is not None:
        boundary_gdf.boundary.plot(ax=ax, linewidth=1, color="black", zorder=1)
    if subs_proj is not None:
        subs_proj.plot(ax=ax, color="blue", markersize=20, marker="s", zorder=3)

    lc = mc.LineCollection(line_segs, colors="limegreen", linewidths=1.5, zorder=2)
    ax.add_collection(lc)

    s0 = storm_data[0]
    storm_fill   = plt.Circle(s0["center"], s0["radius"],
                               facecolor="red", alpha=0.15, zorder=4)
    storm_ring   = plt.Circle(s0["center"], s0["radius"],
                               fill=False, edgecolor="red", linewidth=1.5, zorder=4)
    storm_marker, = ax.plot(*s0["center"], "r*", markersize=12, zorder=5)
    ax.add_patch(storm_fill)
    ax.add_patch(storm_ring)

    net_bounds = lines_proj.total_bounds
    storm_xs   = [s["center"][0] for s in storm_data]
    storm_ys   = [s["center"][1] for s in storm_data]
    margin     = 8000
    ax.set_xlim(min(net_bounds[0], min(storm_xs)) - margin,
                max(net_bounds[2], max(storm_xs)) + margin)
    ax.set_ylim(min(net_bounds[1], min(storm_ys)) - margin,
                max(net_bounds[3], max(storm_ys)) + margin)
    ax.set_aspect("equal")
    ax.set_title("Algorithm 2 — Power Line Failure Assessment", fontsize=13)
    ax.set_xlabel("X  (m, EPSG:32618)")
    ax.set_ylabel("Y  (m, EPSG:32618)")

    info = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=9,
                   bbox=dict(boxstyle="round", facecolor="white", alpha=0.8), zorder=6)

    legend_handles = [
        Line2D([0], [0], color="limegreen", lw=2,  label="Intact line"),
        Line2D([0], [0], color="red",       lw=2,  label="Failed line"),
        Patch(facecolor="red", alpha=0.15, edgecolor="red", label="Storm radius"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="blue",
               markersize=8, label="Substation"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8)

    def update(frame):
        colors = np.where(L_status[frame] == 1, "limegreen", "red")
        lc.set_colors(colors)

        cx, cy = storm_data[frame]["center"]
        for circle in (storm_fill, storm_ring):
            circle.center = (cx, cy)
            circle.set_radius(storm_data[frame]["radius"])
        storm_marker.set_data([cx], [cy])

        n_failed = int(np.sum(L_status[frame] == 0))
        info.set_text(
            f"Hour  : {storm_data[frame]['t']:>3d}\n"
            f"Wind  : {storm_data[frame]['wind_speed']:>5.1f} m/s\n"
            f"Radius: {storm_data[frame]['radius']:>6.0f} m\n"
            f"Failed: {n_failed:>3d} / {n_lines}"
        )
        return lc, storm_fill, storm_ring, storm_marker, info

    anim = FuncAnimation(fig, update, frames=T, interval=interval,
                         blit=False, repeat=repeat)
    plt.tight_layout()
    return anim

# Main function
if __name__ == "__main__":
    storm_data, lines_proj, subs_proj, boundary_gdf = load_data()

    L_status, L_wind = assess_powerline_failures(storm_data, lines_proj)

    anim = animate_failures(
        storm_data, lines_proj, L_status,
        subs_proj=subs_proj,
        boundary_gdf=boundary_gdf,
    )
    plt.show()
