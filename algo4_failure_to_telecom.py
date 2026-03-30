'''
Algorithm 4: Failure Propagation to Telecommunication Network

    Input:
        - Power line status L_status: (T, N_lines) from Algorithm 2 — 1=intact, 0=failed
        - Substation-tower dependency map (built from data)
        - Backup duration h: per-tower backup_hours field
        - Time horizon T: 24 hours

    Output:
        - Tower status T_status: (T, N_towers) — 0=operational, 1=failed

'''

import json
from collections import defaultdict
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.collections as mc
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


def load_data(
    storm_path      = "synthetic_data/storm_data.geojson",
    powerline_path  = "synthetic_data/powerline.geojson",
    substation_path = "synthetic_data/substation.geojson",
    tower_path      = "synthetic_data/tower.geojson",
    boundary_path   = "jsons/Precinct_Boundaries.geojson",
    epsg            = 32618,
):
    """
    Load all spatial data and reproject to a common CRS.

    Returns: 
        - storm_data    : list of dicts  {t, center (x,y), radius, wind_speed}
        - lines_proj    : GeoDataFrame   power lines in projected CRS
        - subs_proj     : GeoDataFrame   substations in projected CRS
        - towers_proj   : GeoDataFrame   telecom towers in projected CRS
        - boundary_gdf  : GeoDataFrame   study boundary in projected CRS

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

    lines_proj   = gpd.read_file(powerline_path).to_crs(epsg=epsg).reset_index(drop=True)
    subs_proj    = gpd.read_file(substation_path).to_crs(epsg=epsg)
    towers_proj  = gpd.read_file(tower_path).to_crs(epsg=epsg)
    boundary_gdf = gpd.read_file(boundary_path).to_crs(epsg=epsg)

    return storm_data, lines_proj, subs_proj, towers_proj, boundary_gdf


# Algorithm 4: Failure Propagation to Telecommunication Network
def assess_telecom_failures(
    L_status,
    lines_gdf,
    towers_gdf,
):
    """
    Algorithm 4: Failure Propagation to Telecommunication Network.

    A tower loses grid power when all power lines connected to its serving
    substation have failed.  It then runs on battery backup for backup_hours
    timesteps; after that it fails permanently.

    Substation power rule:
        substation S is powered at time t  ⟺
        at least one line with from_node=S or to_node=S is intact (L_status=1)

    Parameters:
        - L_status   : np.ndarray (T, N_lines)  — from Algorithm 2 (1=intact, 0=failed)
        - lines_gdf  : GeoDataFrame             — must have from_node, to_node columns
                                                row order must match L_status columns
        - towers_gdf : GeoDataFrame             — must have substation_id, backup_hours

    Returns:
        - T_status      : np.ndarray (T, N_towers)  0=operational, 1=failed
        - backup_elapsed: np.ndarray (T, N_towers)  hours spent on backup at each step

    """
    T         = L_status.shape[0]
    n_towers  = len(towers_gdf)

    # Build substation_id → set of positional line indices
    sub_to_lines = defaultdict(set)
    for idx, row in enumerate(lines_gdf.itertuples()):
        sub_to_lines[row.from_node].add(idx)
        sub_to_lines[row.to_node].add(idx)

    tower_sub    = towers_gdf["substation_id"].values
    tower_backup = towers_gdf["backup_hours"].values.astype(int)

    T_status       = np.zeros((T, n_towers), dtype=int)
    backup_elapsed = np.zeros((T, n_towers), dtype=int)
    _timer         = np.zeros(n_towers, dtype=int)   # running backup timer

    for t in range(T):
        # Determine which substations have grid power at this timestep
        sub_powered = {}
        for sub_id, line_idxs in sub_to_lines.items():
            sub_powered[sub_id] = any(L_status[t, i] == 1 for i in line_idxs)

        for j in range(n_towers):
            if T_status[t, j] == 1:              # already failed — propagate
                _timer[j] = tower_backup[j] + 1
                backup_elapsed[t, j] = _timer[j]
                continue

            sub_id = tower_sub[j]
            if sub_powered.get(sub_id, False):
                _timer[j] = 0                    # grid power restored
            else:
                _timer[j] += 1                   # another hour on backup

            backup_elapsed[t, j] = _timer[j]

            if _timer[j] > tower_backup[j]:
                T_status[t:, j] = 1              # battery exhausted — fails

    return T_status, backup_elapsed


# Visualization and animation
def animate_telecom_failures(
    storm_data,
    lines_proj,
    towers_proj,
    L_status,
    T_status,
    subs_proj    = None,
    boundary_gdf = None,
    interval     = 600,
    repeat       = False,
):
    """
    Animate failure propagation to the telecom network over time.

    Shows:
      - Power line network coloured by L_status (green=intact, grey=failed)
      - Telecom towers coloured by T_status (green=operational, red=failed)
      - Storm circle for spatial context

    Parameters:
        - storm_data   : list of dicts (output of load_data)
        - lines_proj   : GeoDataFrame  projected power lines
        - towers_proj  : GeoDataFrame  projected towers
        - L_status     : np.ndarray    (T, N_lines)  from Algorithm 2
        - T_status     : np.ndarray    (T, N_towers) from assess_telecom_failures
        - subs_proj    : GeoDataFrame  (optional) substations overlay
        - boundary_gdf : GeoDataFrame  (optional) study boundary
        - interval     : int           ms between frames
        - repeat       : bool          loop animation

    Returns:
        - anim : matplotlib FuncAnimation

    """
    T        = len(storm_data)
    n_towers = len(towers_proj)
    n_lines  = len(lines_proj)

    line_segs  = [list(geom.coords) for geom in lines_proj.geometry]
    tower_x    = towers_proj.geometry.x.values
    tower_y    = towers_proj.geometry.y.values

    fig, ax = plt.subplots(figsize=(11, 9))

    if boundary_gdf is not None:
        boundary_gdf.boundary.plot(ax=ax, linewidth=1, color="black", zorder=1)

    # Power line layer (background)
    lc = mc.LineCollection(line_segs, colors="limegreen", linewidths=1.2, zorder=2)
    ax.add_collection(lc)

    # Substation markers (static)
    if subs_proj is not None:
        subs_proj.plot(ax=ax, color="goldenrod", markersize=15, marker="s", zorder=3)

    # Tower scatter
    tower_colors = np.where(T_status[0] == 1, "red", "limegreen")
    sc = ax.scatter(tower_x, tower_y,
                    c=tower_colors, s=40, marker="^", zorder=4,
                    edgecolors="black", linewidths=0.4)

    # Storm circle
    s0 = storm_data[0]
    storm_fill   = plt.Circle(s0["center"], s0["radius"],
                               facecolor="steelblue", alpha=0.12, zorder=5)
    storm_ring   = plt.Circle(s0["center"], s0["radius"],
                               fill=False, edgecolor="steelblue", linewidth=1.5, zorder=5)
    storm_marker, = ax.plot(*s0["center"], "b*", markersize=12, zorder=6)
    ax.add_patch(storm_fill)
    ax.add_patch(storm_ring)

    # Axis limits
    all_x = list(tower_x) + [s["center"][0] for s in storm_data]
    all_y = list(tower_y) + [s["center"][1] for s in storm_data]
    net_b = lines_proj.total_bounds
    margin = 8000
    ax.set_xlim(min(net_b[0], min(all_x)) - margin,
                max(net_b[2], max(all_x)) + margin)
    ax.set_ylim(min(net_b[1], min(all_y)) - margin,
                max(net_b[3], max(all_y)) + margin)
    ax.set_aspect("equal")
    ax.set_title("Algorithm 4 — Failure Propagation to Telecom Network", fontsize=13)
    ax.set_xlabel("X  (m, EPSG:32618)")
    ax.set_ylabel("Y  (m, EPSG:32618)")

    info = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=9,
                   bbox=dict(boxstyle="round", facecolor="white", alpha=0.8), zorder=7)

    legend_handles = [
        Line2D([0], [0], color="limegreen", lw=2,          label="Intact power line"),
        Line2D([0], [0], color="dimgrey",   lw=2,          label="Failed power line"),
        Line2D([0], [0], marker="^", color="w",
               markerfacecolor="limegreen", markersize=9,
               markeredgecolor="black",    label="Tower operational"),
        Line2D([0], [0], marker="^", color="w",
               markerfacecolor="red",      markersize=9,
               markeredgecolor="black",    label="Tower failed"),
        Patch(facecolor="goldenrod", edgecolor="none", label="Substation"),
        Patch(facecolor="steelblue", alpha=0.3, edgecolor="steelblue",
              label="Storm radius"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8)

    def update(frame):
        # Update power lines
        line_colors = np.where(L_status[frame] == 1, "limegreen", "dimgrey")
        lc.set_colors(line_colors)

        # Update towers
        t_colors = np.where(T_status[frame] == 1, "red", "limegreen")
        sc.set_facecolors(t_colors)

        # Update storm
        cx, cy = storm_data[frame]["center"]
        for circle in (storm_fill, storm_ring):
            circle.center = (cx, cy)
            circle.set_radius(storm_data[frame]["radius"])
        storm_marker.set_data([cx], [cy])

        n_line_failed  = int(np.sum(L_status[frame] == 0))
        n_tower_failed = int(np.sum(T_status[frame] == 1))
        info.set_text(
            f"Hour         : {storm_data[frame]['t']:>3d}\n"
            f"Wind         : {storm_data[frame]['wind_speed']:>5.1f} m/s\n"
            f"Lines failed : {n_line_failed:>3d} / {n_lines}\n"
            f"Towers failed: {n_tower_failed:>3d} / {n_towers}"
        )
        return lc, sc, storm_fill, storm_ring, storm_marker, info

    anim = FuncAnimation(fig, update, frames=T, interval=interval,
                         blit=False, repeat=repeat)
    plt.tight_layout()
    return anim


# Main
if __name__ == "__main__":
    from algo2_powerline_failure import assess_powerline_failures as _alg2 
    storm_data, lines_proj, subs_proj, towers_proj, boundary_gdf = load_data()

    # Algorithm 2 — power line failures (input)
    L_status, _ = _alg2(storm_data, lines_proj)

    # Algorithm 4 — telecom failure propagation
    T_status, backup_elapsed = assess_telecom_failures(L_status, lines_proj, towers_proj)

    anim = animate_telecom_failures(
        storm_data, lines_proj, towers_proj,
        L_status, T_status,
        subs_proj=subs_proj,
        boundary_gdf=boundary_gdf,
    )
    plt.show()
