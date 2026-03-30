'''
Algorithm 3: Road Blockage Assessment

    Input:
        - Storm data S_df: use a synthetic data
        - Road network G_roads with tree counts: use a synthetic data
        - Fragility parameters (μ_T, σ_T): use synthetic parameters
        - Time horizon T: 24 hours
    Output:
        - Road status R_status: (T, N) array — 0 = clear, 1 = blocked
        - Blockage probabilities R_prob: (T, N) array

'''

import json
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.collections as mc
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Patch, Polygon as MplPolygon
from math import log, erfc, sqrt


# Fragility function for trees — same form as eq. (4) but with tree parameters
def lognormal_cdf_tree(x, mu, sigma):
    """
        Lognormal CDF for tree fragility.

        P(tree falls | wind speed x) = Φ((ln x − μ_T) / σ_T)

        Parameters:
            - x     : local wind speed at the road centroid
            - mu    : ln-mean of tree fragility (μ_T, Table II)
            - sigma : ln-std of tree fragility (σ_T, Table II)
        Returns:
            - p_fail : probability that a single tree falls
    """
    if x <= 0:
        return 0.0
    z = (log(x) - mu) / (sigma * sqrt(2))
    return 0.5 * erfc(-z)


def load_data(
    storm_path    = "synthetic_data/storm_data.geojson",
    roads_path    = "synthetic_data/roads_with_trees.geojson",
    boundary_path = "jsons/Precinct_Boundaries.geojson",
    epsg          = 32618,
):
    """
        Load storm and road data and reproject to a common CRS.

        Returns:
        - storm_data   : list of dicts  {t, center (x,y), radius, wind_speed}
        - roads_proj   : GeoDataFrame   roads in projected CRS, with cent_x / cent_y columns
        - boundary_gdf : GeoDataFrame   study boundary in projected CRS

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

    roads_proj   = gpd.read_file(roads_path).to_crs(epsg=epsg)
    boundary_gdf = gpd.read_file(boundary_path).to_crs(epsg=epsg)

    # Pre-compute centroids in projected CRS for fast distance calculations
    roads_proj = roads_proj.copy()
    roads_proj["cent_x"] = roads_proj.geometry.centroid.x
    roads_proj["cent_y"] = roads_proj.geometry.centroid.y

    return storm_data, roads_proj, boundary_gdf


# Algorithm 3: Road Blockage Assessment
def assess_road_blockages(
    storm_data,
    roads_gdf,
    mu_t    = 3.50,   # ln-mean  (Table II: Tree Fragility Mean)
    sigma_t = 0.30,   # ln-std   (Table II: Tree Fragility Std)
    seed    = 42,
):
    
    """
        Algorithm 3: Road Blockage Assessment.

        A road is blocked when at least one of its roadside trees falls onto it.
        The probability of blocking given n trees and local wind speed w is:

            P(block) = 1 − (1 − P_tree_fail(w))^n

        where P_tree_fail uses the lognormal fragility curve with (μ_T, σ_T).
        Blockage is permanent once triggered (no restoration in this algorithm).

        Parameters:
            - storm_data : list of dicts  {t, center, radius, wind_speed}
            - roads_gdf  : GeoDataFrame   must have cent_x, cent_y, tree_count columns
            - mu_t, sigma_t : lognormal fragility parameters for trees
            - seed       : random seed for reproducibility

        Returns:
            - R_status : np.ndarray shape (T, N)  0 = clear, 1 = blocked
            - R_prob   : np.ndarray shape (T, N)  blockage probability at each step

    """
    T       = len(storm_data)
    n_roads = len(roads_gdf)
    cent_x  = roads_gdf["cent_x"].values.astype(float)
    cent_y  = roads_gdf["cent_y"].values.astype(float)
    n_trees = roads_gdf["tree_count"].values.astype(int)

    R_status = np.zeros((T, n_roads), dtype=int)   # 0 = clear
    R_prob   = np.zeros((T, n_roads))
    rng      = np.random.default_rng(seed)

    for t, storm in enumerate(storm_data):
        cx, cy = storm["center"]
        r      = storm["radius"]
        w_gust = storm["wind_speed"]

        # Carry forward roads already blocked in previous timestep
        if t > 0:
            R_status[t] = R_status[t - 1]

        dist   = np.sqrt((cent_x - cx) ** 2 + (cent_y - cy) ** 2)
        inside = np.where(dist < r)[0]

        for i in inside:
            if R_status[t, i] == 1:          # already blocked — skip
                R_prob[t, i] = 1.0
                continue

            n = n_trees[i]
            if n == 0:
                continue

            # Distance-based wind decay: 100% at center, 50% at edge
            decay   = 1.0 - 0.5 * dist[i] / r
            w_local = w_gust * decay

            p_tree  = lognormal_cdf_tree(w_local, mu_t, sigma_t)
            p_block = 1.0 - (1.0 - p_tree) ** n   # at least one tree falls
            R_prob[t, i] = p_block

            if rng.random() < p_block:
                R_status[t:, i] = 1             # blocked — remains blocked

    return R_status, R_prob


# Visualization and animation
def animate_blockages(
    storm_data,
    roads_proj,
    R_status,
    boundary_gdf = None,
    interval     = 600,
    repeat       = False,
):
    """
        Animate the road blockage assessment over time.

        Roads are rendered as scatter dots at their centroids for performance
        (10 k+ polygon segments are too heavy to recolor as patches every frame).

        Parameters:
            - storm_data   : list of dicts (output of load_data)
            - roads_proj   : GeoDataFrame  projected roads with cent_x / cent_y
            - R_status     : np.ndarray    (T, N) output of assess_road_blockages
            - boundary_gdf : GeoDataFrame  (optional) study boundary overlay
            - interval     : int           milliseconds between frames
            - repeat       : bool          whether to loop the animation

        Returns:
            - anim : matplotlib FuncAnimation

    """
    T       = len(storm_data)
    n_roads = len(roads_proj)

    # Build MplPolygon patches — handle Polygon and MultiPolygon
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

    s0 = storm_data[0]
    storm_fill   = plt.Circle(s0["center"], s0["radius"],
                               facecolor="steelblue", alpha=0.15, zorder=3)
    storm_ring   = plt.Circle(s0["center"], s0["radius"],
                               fill=False, edgecolor="steelblue", linewidth=1.5, zorder=3)
    storm_marker, = ax.plot(*s0["center"], "b*", markersize=12, zorder=4)
    ax.add_patch(storm_fill)
    ax.add_patch(storm_ring)

    bounds    = roads_proj.total_bounds
    storm_xs  = [s["center"][0] for s in storm_data]
    storm_ys  = [s["center"][1] for s in storm_data]
    margin    = 5000
    ax.set_xlim(min(bounds[0], min(storm_xs)) - margin,
                max(bounds[2], max(storm_xs)) + margin)
    ax.set_ylim(min(bounds[1], min(storm_ys)) - margin,
                max(bounds[3], max(storm_ys)) + margin)
    ax.set_aspect("equal")
    ax.set_title("Algorithm 3 — Road Blockage Assessment", fontsize=13)
    ax.set_xlabel("X  (m, EPSG:32618)")
    ax.set_ylabel("Y  (m, EPSG:32618)")

    info = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=9,
                   bbox=dict(boxstyle="round", facecolor="white", alpha=0.8), zorder=5)

    legend_handles = [
        Patch(facecolor="limegreen", edgecolor="none", label="Clear road"),
        Patch(facecolor="red",       edgecolor="none", label="Blocked road"),
        Patch(facecolor="steelblue", alpha=0.3, edgecolor="steelblue",
              label="Storm radius"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8)

    def update(frame):
        colors = np.where(R_status[frame] == 1, "red", "limegreen")
        pc.set_facecolors(colors)

        cx, cy = storm_data[frame]["center"]
        for circle in (storm_fill, storm_ring):
            circle.center = (cx, cy)
            circle.set_radius(storm_data[frame]["radius"])
        storm_marker.set_data([cx], [cy])

        n_blocked = int(np.sum(R_status[frame] == 1))
        info.set_text(
            f"Hour   : {storm_data[frame]['t']:>3d}\n"
            f"Wind   : {storm_data[frame]['wind_speed']:>5.1f} m/s\n"
            f"Radius : {storm_data[frame]['radius']:>6.0f} m\n"
            f"Blocked: {n_blocked:>4d} / {n_roads}"
        )
        return pc, storm_fill, storm_ring, storm_marker, info

    anim = FuncAnimation(fig, update, frames=T, interval=interval,
                         blit=False, repeat=repeat)
    plt.tight_layout()
    return anim


# Main
if __name__ == "__main__":
    storm_data, roads_proj, boundary_gdf = load_data()

    R_status, R_prob = assess_road_blockages(storm_data, roads_proj)

    anim = animate_blockages(
        storm_data, roads_proj, R_status,
        boundary_gdf=boundary_gdf,
    )
    plt.show()
