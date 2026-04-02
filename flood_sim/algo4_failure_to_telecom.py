'''
Algorithm 4 (Flood): Failure Propagation to Telecommunication Network

    Input:
        - Power line status L_status: (T, N_lines) from Algorithm 2 — 1=intact, 0=failed
        - Substation-tower dependency map (built from data)
        - Backup duration h: per-tower backup_hours field
        - Time horizon T: 24 hours

    Output:
        - Tower status T_status: (T, N_towers) — 0=operational, 1=failed

    Note: The core failure propagation logic is hazard-agnostic — it only
    depends on L_status from Algorithm 2. The flood context is reflected in
    load_data (reads flood_data.geojson) and the animation (expanding flood
    circle instead of a moving storm).
'''

import json
from collections import defaultdict
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.collections as mc
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Patch, Circle
from matplotlib.lines import Line2D


def load_data(
    flood_path      = "synthetic_data/flood_data.geojson",
    powerline_path  = "synthetic_data/powerline.geojson",
    substation_path = "synthetic_data/substation.geojson",
    tower_path      = "synthetic_data/tower.geojson",
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
    towers_proj  : GeoDataFrame   telecom towers in projected CRS
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

    lines_proj   = gpd.read_file(powerline_path).to_crs(epsg=epsg).reset_index(drop=True)
    subs_proj    = gpd.read_file(substation_path).to_crs(epsg=epsg)
    towers_proj  = gpd.read_file(tower_path).to_crs(epsg=epsg)
    boundary_gdf = gpd.read_file(boundary_path).to_crs(epsg=epsg)

    return flood_data, lines_proj, subs_proj, towers_proj, boundary_gdf


# Algorithm 4: Failure Propagation to Telecommunication Network
def assess_telecom_failures(
    L_status,
    lines_gdf,
    towers_gdf,
):
    """
    Algorithm 4: Failure Propagation to Telecommunication Network.

    Hazard-agnostic — only depends on L_status from Algorithm 2.

    A tower loses grid power when all power lines connected to its serving
    substation have failed. It then runs on battery backup for backup_hours
    timesteps; after that it fails permanently.

    Substation power rule:
        substation S is powered at time t  ⟺
        at least one line with from_node=S or to_node=S is intact (L_status=1)

    Parameters
    ----------
    L_status   : np.ndarray (T, N_lines)  from Algorithm 2 (1=intact, 0=failed)
    lines_gdf  : GeoDataFrame             must have from_node, to_node columns
    towers_gdf : GeoDataFrame             must have substation_id, backup_hours

    Returns
    -------
    T_status       : np.ndarray (T, N_towers)  0=operational, 1=failed
    backup_elapsed : np.ndarray (T, N_towers)  hours spent on backup at each step
    """
    T        = L_status.shape[0]
    n_towers = len(towers_gdf)

    # Build substation_id → set of positional line indices
    sub_to_lines = defaultdict(set)
    for idx, row in enumerate(lines_gdf.itertuples()):
        sub_to_lines[row.from_node].add(idx)
        sub_to_lines[row.to_node].add(idx)

    tower_sub    = towers_gdf["substation_id"].values
    tower_backup = towers_gdf["backup_hours"].values.astype(int)

    T_status       = np.zeros((T, n_towers), dtype=int)
    backup_elapsed = np.zeros((T, n_towers), dtype=int)
    _timer         = np.zeros(n_towers, dtype=int)

    for t in range(T):
        # Determine which substations have grid power at this timestep
        sub_powered = {}
        for sub_id, line_idxs in sub_to_lines.items():
            sub_powered[sub_id] = any(L_status[t, i] == 1 for i in line_idxs)

        for j in range(n_towers):
            if T_status[t, j] == 1:             # already failed — propagate
                _timer[j] = tower_backup[j] + 1
                backup_elapsed[t, j] = _timer[j]
                continue

            sub_id = tower_sub[j]
            if sub_powered.get(sub_id, False):
                _timer[j] = 0                   # grid power restored
            else:
                _timer[j] += 1                  # another hour on backup

            backup_elapsed[t, j] = _timer[j]

            if _timer[j] > tower_backup[j]:
                T_status[t:, j] = 1             # battery exhausted — fails

    return T_status, backup_elapsed


# Visualization and animation
def animate_telecom_failures(
    flood_data,
    lines_proj,
    towers_proj,
    L_status,
    T_status,
    subs_proj    = None,
    boundary_gdf = None,
    n_frames     = 120,
    interval     = 150,
    repeat       = False,
):
    """
    Animate failure propagation to the telecom network over time.

    Animation frames are interpolated across the 24-hour flood_data so the
    flood circle expands smoothly (same approach as algo2_powerline_failure).

    Shows:
      - Power line network coloured by L_status (green=intact, grey=failed)
      - Telecom towers coloured by T_status (green=operational, red=failed)
      - Expanding flood circle for spatial context

    Parameters
    ----------
    flood_data   : list of dicts (output of load_data)
    lines_proj   : GeoDataFrame  projected power lines
    towers_proj  : GeoDataFrame  projected towers
    L_status     : np.ndarray    (T, N_lines)  from Algorithm 2
    T_status     : np.ndarray    (T, N_towers) from assess_telecom_failures
    subs_proj    : GeoDataFrame  (optional) substations overlay
    boundary_gdf : GeoDataFrame  (optional) study boundary
    n_frames     : int           total animation frames
    interval     : int           ms between frames
    repeat       : bool          loop animation
    """
    T        = len(flood_data)
    n_towers = len(towers_proj)
    n_lines  = len(lines_proj)

    # Interpolate flood radius and depth across n_frames
    data_t   = np.array([s["t"]            for s in flood_data], dtype=float)
    radii    = np.array([s["flood_radius"]  for s in flood_data], dtype=float)
    depths   = np.array([s["max_depth"]     for s in flood_data], dtype=float)
    hours    = np.linspace(data_t[0], data_t[-1], n_frames)
    r_interp = np.interp(hours, data_t, radii)
    d_interp = np.interp(hours, data_t, depths)

    # Map each animation frame to the nearest data timestep for L/T_status
    t_indices = np.round(np.linspace(0, T - 1, n_frames)).astype(int)

    origin    = flood_data[0]["origin"]
    line_segs = [list(geom.coords) for geom in lines_proj.geometry]
    tower_x   = towers_proj.geometry.x.values
    tower_y   = towers_proj.geometry.y.values

    fig, ax = plt.subplots(figsize=(11, 9))

    if boundary_gdf is not None:
        boundary_gdf.boundary.plot(ax=ax, linewidth=1, color="black", zorder=1)

    # Power line layer
    lc = mc.LineCollection(line_segs, colors="limegreen", linewidths=1.2, zorder=2)
    ax.add_collection(lc)

    # Substation markers (static)
    if subs_proj is not None:
        subs_proj.plot(ax=ax, color="goldenrod", markersize=15, marker="s", zorder=3)

    # Tower scatter
    sc = ax.scatter(tower_x, tower_y,
                    c=["limegreen"] * n_towers, s=40, marker="^", zorder=4,
                    edgecolors="black", linewidths=0.4)

    # Flood circle
    flood_fill = Circle(origin, 0, color="steelblue", alpha=0.15, zorder=5)
    flood_ring = Circle(origin, 0, fill=False, edgecolor="steelblue", linewidth=1.5, zorder=5)
    ax.add_patch(flood_fill)
    ax.add_patch(flood_ring)
    ax.plot(*origin, "r^", markersize=10, zorder=6, label="Flood origin")

    # Axis limits
    net_b  = lines_proj.total_bounds
    margin = 8000
    ax.set_xlim(net_b[0] - margin, net_b[2] + margin)
    ax.set_ylim(net_b[1] - margin, net_b[3] + margin)
    ax.set_aspect("equal")
    ax.set_title("Algorithm 4 (Flood) — Failure Propagation to Telecom Network", fontsize=13)
    ax.set_xlabel("X  (m, EPSG:32618)")
    ax.set_ylabel("Y  (m, EPSG:32618)")

    info = ax.text(
        0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8), zorder=7,
    )

    legend_handles = [
        Line2D([0], [0], color="limegreen", lw=2,          label="Intact power line"),
        Line2D([0], [0], color="dimgrey",   lw=2,          label="Failed power line"),
        Line2D([0], [0], marker="^", color="w",
               markerfacecolor="limegreen", markersize=9,
               markeredgecolor="black",     label="Tower operational"),
        Line2D([0], [0], marker="^", color="w",
               markerfacecolor="red",       markersize=9,
               markeredgecolor="black",     label="Tower failed"),
        Patch(facecolor="goldenrod",  edgecolor="none",      label="Substation"),
        Patch(facecolor="steelblue",  alpha=0.2,
              edgecolor="steelblue",                         label="Flood extent"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8)

    def update(frame):
        t_idx = t_indices[frame]

        # Power lines
        line_colors = np.where(L_status[t_idx] == 1, "limegreen", "dimgrey")
        lc.set_colors(line_colors)

        # Towers
        t_colors = np.where(T_status[t_idx] == 1, "red", "limegreen")
        sc.set_facecolors(t_colors)

        # Flood circle
        r = r_interp[frame]
        flood_fill.set_radius(r)
        flood_ring.set_radius(r)

        n_line_failed  = int(np.sum(L_status[t_idx] == 0))
        n_tower_failed = int(np.sum(T_status[t_idx] == 1))
        info.set_text(
            f"Hour         : {hours[frame]:>5.1f}\n"
            f"Flood radius : {r/1000:>6.2f} km\n"
            f"Depth origin : {d_interp[frame]:>5.2f} m\n"
            f"Lines failed : {n_line_failed:>3d} / {n_lines}\n"
            f"Towers failed: {n_tower_failed:>3d} / {n_towers}"
        )
        return lc, sc, flood_fill, flood_ring, info

    anim = FuncAnimation(fig, update, frames=n_frames,
                         interval=interval, blit=False, repeat=repeat)
    plt.tight_layout()
    return anim


if __name__ == "__main__":
    from algo2_powerline_failure import assess_powerline_failures as _alg2

    flood_data, lines_proj, subs_proj, towers_proj, boundary_gdf = load_data()

    # Algorithm 2 — power line failures (input to Algorithm 4)
    L_status, _ = _alg2(flood_data, lines_proj)

    # Algorithm 4 — telecom failure propagation
    T_status, backup_elapsed = assess_telecom_failures(L_status, lines_proj, towers_proj)

    anim = animate_telecom_failures(
        flood_data, lines_proj, towers_proj,
        L_status, T_status,
        subs_proj    = subs_proj,
        boundary_gdf = boundary_gdf,
    )
    plt.show()
