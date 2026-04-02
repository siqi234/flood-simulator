'''
Algorithm 3 (Flood): Road Blockage Assessment

    Input:
        - Flood data F_df: synthetic_data/flood_data.geojson
        - Road network G_roads: synthetic_data/roads_with_trees.geojson
        - Fragility parameters (μ, σ): lognormal, calibrated to flood depth
        - Time horizon T: 24 hours

    Output:
        - R_status : np.ndarray (T, N)  0 = clear, 1 = blocked
        - R_prob   : np.ndarray (T, N)  blockage probability at each timestep

    Key difference from windstorm version:
        - Hazard is flood depth (m), not wind speed (m/s)
        - No tree count — road blockage is caused directly by inundation
        - Flood zone is persistent: once a road is flooded it stays flooded
        - All flooded roads are re-evaluated every timestep (continued submersion)
'''

import json
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.collections as mc
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Patch, Circle
from math import log, erfc, sqrt


# Fragility function — same lognormal CDF form, x is now flood depth (m)
def lognormal_cdf_road(x, mu, sigma):
    """
    Lognormal CDF fragility for road inundation.

    P(road blocked | flood depth x)

    Parameters
    ----------
    x     : local flood depth at the road centroid (metres)
    mu    : ln-mean  (50% blockage probability at exp(mu) metres depth)
    sigma : ln-std

    Returns
    -------
    p_block : probability the road is blocked at depth x
    """
    if x <= 0:
        return 0.0
    z = (log(x) - mu) / (sigma * sqrt(2))
    return 0.5 * erfc(-z)


def load_data(
    flood_path    = "synthetic_data/flood_data.geojson",
    roads_path    = "synthetic_data/roads_with_trees.geojson",
    boundary_path = "jsons/Precinct_Boundaries.geojson",
    epsg          = 32618,
):
    """
    Load flood and road data and reproject to a common CRS.

    Returns
    -------
    flood_data   : list of dicts  {t, origin (x,y), flood_radius, max_depth}
    roads_proj   : GeoDataFrame   roads in projected CRS with cent_x, cent_y
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

    roads_proj   = gpd.read_file(roads_path).to_crs(epsg=epsg)
    boundary_gdf = gpd.read_file(boundary_path).to_crs(epsg=epsg)

    # Pre-compute centroids for fast distance calculations
    roads_proj = roads_proj.copy()
    roads_proj["cent_x"] = roads_proj.geometry.centroid.x
    roads_proj["cent_y"] = roads_proj.geometry.centroid.y

    return flood_data, roads_proj, boundary_gdf


# Algorithm 3 (Flood): Road Blockage Assessment
def assess_road_blockages(
    flood_data,
    roads_gdf,
    mu    = -0.92,  # ln-mean: 50% blockage at ~0.4m depth  (ln(0.4) ≈ -0.92)
    sigma = 0.30,   # ln-std
    seed  = 42,
):
    """
    Algorithm 3 (Flood): Road Blockage Assessment.

    A road is blocked when flood depth at its centroid exceeds its tolerance.
    Depth decays linearly from max_depth at the origin to 0 at the flood edge.

    Persistent flood logic (same as algo2):
        - flooded_mask tracks which roads are currently inundated
        - Once flooded, a road stays in the flood zone for all subsequent steps
        - All flooded intact roads are re-evaluated each timestep

    Parameters
    ----------
    flood_data : list of dicts  {t, origin, flood_radius, max_depth}
    roads_gdf  : GeoDataFrame   must have cent_x, cent_y columns
    mu, sigma  : lognormal fragility parameters (depth-based)
    seed       : random seed for reproducibility

    Returns
    -------
    R_status : np.ndarray (T, N)  0 = clear, 1 = blocked
    R_prob   : np.ndarray (T, N)  blockage probability at each timestep
    """
    T       = len(flood_data)
    n_roads = len(roads_gdf)
    cent_x  = roads_gdf["cent_x"].values.astype(float)
    cent_y  = roads_gdf["cent_y"].values.astype(float)

    R_status     = np.zeros((T, n_roads), dtype=int)   # 0 = clear
    R_prob       = np.zeros((T, n_roads))
    flooded_mask = np.zeros(n_roads, dtype=bool)        # persistent
    rng          = np.random.default_rng(seed)

    for t, flood in enumerate(flood_data):
        ox, oy       = flood["origin"]
        flood_radius = flood["flood_radius"]
        max_depth    = flood["max_depth"]

        # Carry forward roads already blocked in previous timestep
        if t > 0:
            R_status[t] = R_status[t - 1]

        if flood_radius == 0:
            continue

        # Distance from each road centroid to flood origin
        dist = np.sqrt((cent_x - ox) ** 2 + (cent_y - oy) ** 2)

        # Update persistent flood mask
        flooded_mask |= dist < flood_radius

        # Re-evaluate all currently flooded roads
        flooded_indices = np.where(flooded_mask)[0]
        for i in flooded_indices:
            if R_status[t, i] == 1:         # already blocked — skip
                R_prob[t, i] = 1.0
                continue

            depth        = max_depth * max(0.0, 1.0 - dist[i] / flood_radius)
            p_block      = lognormal_cdf_road(depth, mu, sigma)
            R_prob[t, i] = p_block

            if rng.random() < p_block:
                R_status[t:, i] = 1         # blocked — remains blocked

    return R_status, R_prob


# Visualization and animation
def animate_blockages(
    flood_data,
    roads_proj,
    R_status,
    boundary_gdf = None,
    n_frames     = 120,
    interval     = 150,
    repeat       = False,
):
    """
    Animate the road blockage assessment over time.

    Animation frames are interpolated across the 24-hour flood_data so the
    flood circle expands smoothly (same approach as algo2_powerline_failure).

    Parameters
    ----------
    flood_data   : list of dicts (output of load_data)
    roads_proj   : GeoDataFrame  projected roads with cent_x, cent_y
    R_status     : np.ndarray    (T, N) output of assess_road_blockages
    boundary_gdf : GeoDataFrame  (optional) study boundary overlay
    n_frames     : int           total animation frames
    interval     : int           ms between frames
    repeat       : bool          loop animation
    """
    from matplotlib.patches import Polygon as MplPolygon

    T       = len(flood_data)
    n_roads = len(roads_proj)

    # Interpolate flood radius and depth across n_frames
    data_t   = np.array([s["t"]            for s in flood_data], dtype=float)
    radii    = np.array([s["flood_radius"]  for s in flood_data], dtype=float)
    depths   = np.array([s["max_depth"]     for s in flood_data], dtype=float)
    hours    = np.linspace(data_t[0], data_t[-1], n_frames)
    r_interp = np.interp(hours, data_t, radii)
    d_interp = np.interp(hours, data_t, depths)

    # Map each animation frame to the nearest data timestep for R_status
    t_indices = np.round(np.linspace(0, T - 1, n_frames)).astype(int)

    origin = flood_data[0]["origin"]

    # Build road patches — handle Polygon and MultiPolygon
    patches = []
    for geom in roads_proj.geometry:
        if geom.geom_type == "MultiPolygon":
            poly = max(geom.geoms, key=lambda p: p.area)
        else:
            poly = geom
        patches.append(MplPolygon(list(poly.exterior.coords)))

    fig, ax = plt.subplots(figsize=(11, 9))

    if boundary_gdf is not None:
        boundary_gdf.boundary.plot(ax=ax, linewidth=1, color="black", zorder=1)

    pc = mc.PatchCollection(patches, facecolors="limegreen", edgecolors="none", zorder=2)
    ax.add_collection(pc)

    # Flood circle
    flood_fill = Circle(origin, 0, color="steelblue", alpha=0.20, zorder=3)
    flood_ring = Circle(origin, 0, fill=False, edgecolor="steelblue", linewidth=1.5, zorder=3)
    ax.add_patch(flood_fill)
    ax.add_patch(flood_ring)
    ax.plot(*origin, "r^", markersize=10, zorder=4, label="Flood origin")

    bounds = roads_proj.total_bounds
    margin = 5000
    ax.set_xlim(bounds[0] - margin, bounds[2] + margin)
    ax.set_ylim(bounds[1] - margin, bounds[3] + margin)
    ax.set_aspect("equal")
    ax.set_title("Algorithm 3 (Flood) — Road Blockage Assessment", fontsize=13)
    ax.set_xlabel("X  (m, EPSG:32618)")
    ax.set_ylabel("Y  (m, EPSG:32618)")

    info = ax.text(
        0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8), zorder=5,
    )

    legend_handles = [
        Patch(facecolor="limegreen", edgecolor="none", label="Clear road"),
        Patch(facecolor="red",       edgecolor="none", label="Blocked road"),
        Patch(facecolor="steelblue", alpha=0.2,
              edgecolor="steelblue",                   label="Flood extent"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8)

    def update(frame):
        t_idx  = t_indices[frame]
        colors = np.where(R_status[t_idx] == 1, "red", "limegreen")
        pc.set_facecolors(colors)

        r = r_interp[frame]
        flood_fill.set_radius(r)
        flood_ring.set_radius(r)

        n_blocked = int(np.sum(R_status[t_idx] == 1))
        info.set_text(
            f"Hour        : {hours[frame]:>5.1f}\n"
            f"Flood radius: {r/1000:>6.2f} km\n"
            f"Depth origin: {d_interp[frame]:>5.2f} m\n"
            f"Blocked     : {n_blocked:>4d} / {n_roads}"
        )
        return pc, flood_fill, flood_ring, info

    anim = FuncAnimation(fig, update, frames=n_frames,
                         interval=interval, blit=False, repeat=repeat)
    plt.tight_layout()
    return anim


if __name__ == "__main__":
    flood_data, roads_proj, boundary_gdf = load_data()

    R_status, R_prob = assess_road_blockages(flood_data, roads_proj)

    anim = animate_blockages(
        flood_data, roads_proj, R_status,
        boundary_gdf=boundary_gdf,
    )
    plt.show()
