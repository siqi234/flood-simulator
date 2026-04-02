'''
Flood Data Generator

Generates synthetic flood data (radially expanding from a fixed origin)
and saves it as synthetic_data/flood_data.geojson.

Each timestep records:
    - origin      : fixed flood source (x, y) in projected CRS
    - flood_radius: how far the flood has spread (metres) — monotonically grows
    - max_depth   : water depth at the origin (m) — rises then plateaus
'''

import json
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle


def create_flood_data(
    boundary_path   = "jsons/Precinct_Boundaries.geojson",
    hours           = 24,
    epsg            = 32618,
    peak_depth_m    = 1.5,    # max water depth at origin (metres)
    flood_fraction  = 0.75,   # fraction of hours spent expanding (rest = plateau)
):
    """
    Generate synthetic radial flood data over `hours` timesteps.

    The flood origin is fixed at the right-centre edge of the study area
    (matching the convention in radial_water_sim.py).

    Flood radius grows with a cosine S-curve up to `flood_fraction * hours`,
    then stays at max_radius for the remaining timesteps.

    Water depth at origin rises in the same S-curve pattern and plateaus.

    Parameters
    ----------
    boundary_path  : path to boundary GeoJSON
    hours          : number of timesteps
    epsg           : projected CRS (must match powerline data)
    peak_depth_m   : maximum water depth at the flood origin
    flood_fraction : fraction of the time horizon spent expanding

    Returns
    -------
    flood_data : list of dicts {t, origin, flood_radius, max_depth}
    gdf_proj   : projected boundary GeoDataFrame
    """

    gdf  = gpd.read_file(boundary_path)
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    gdf_proj = gdf.to_crs(epsg=epsg)

    minx, miny, maxx, maxy = gdf_proj.union_all().bounds

    # Fixed origin: right edge, vertical centre — same as radial_water_sim.py
    origin = (maxx, (miny + maxy) / 2.0)

    # Max radius: diagonal of bounding box (guarantees full coverage)
    max_radius = np.sqrt((maxx - minx) ** 2 + (maxy - miny) ** 2)

    expand_steps = int(hours * flood_fraction)   # timesteps while expanding
    flood_data   = []

    for t in range(hours):
        if t < expand_steps:
            progress       = t / (expand_steps - 1) if expand_steps > 1 else 1.0
            smooth         = 0.5 - 0.5 * np.cos(np.pi * progress)  # S-curve [0, 1]
            flood_radius   = max_radius * smooth
            depth          = peak_depth_m * smooth
        else:
            # Plateau: flood stays at max extent
            flood_radius   = max_radius
            depth          = peak_depth_m

        flood_data.append({
            "t":            t,
            "origin":       origin,
            "flood_radius": float(flood_radius),
            "max_depth":    float(depth),
        })

    return flood_data, gdf_proj


def save_flood_data(flood_data, out_path="synthetic_data/flood_data.geojson"):
    """Save flood_data list to a GeoJSON file."""

    features = []
    for step in flood_data:
        features.append({
            "type": "Feature",
            "geometry": {
                "type":        "Point",
                "coordinates": list(step["origin"]),
            },
            "properties": {
                "t":            step["t"],
                "flood_radius": step["flood_radius"],
                "max_depth":    step["max_depth"],
            },
        })

    geojson = {"type": "FeatureCollection", "features": features}
    with open(out_path, "w") as f:
        json.dump(geojson, f, indent=2)

    print(f"Saved {len(features)} timesteps → {out_path}")


def animate_flood(flood_data, boundary_gdf, n_frames=120, interval=150, repeat=False):
    """
    Animate the radially expanding flood over the study boundary.

    Animation frames are decoupled from data timesteps: `n_frames` frames are
    interpolated smoothly across the 24-hour flood_data, so the animation is
    fluid regardless of how many hourly data points exist.

    Parameters
    ----------
    flood_data   : list of dicts (output of create_flood_data)
    boundary_gdf : projected GeoDataFrame
    n_frames     : total animation frames (more = smoother, default 120)
    interval     : milliseconds between frames (default 150 → ~18 s total)
    repeat       : whether to loop
    """

    # Pre-interpolate radius and depth across n_frames
    data_t   = np.array([s["t"]            for s in flood_data], dtype=float)
    radii    = np.array([s["flood_radius"]  for s in flood_data], dtype=float)
    depths   = np.array([s["max_depth"]     for s in flood_data], dtype=float)
    hours    = np.linspace(data_t[0], data_t[-1], n_frames)
    r_interp = np.interp(hours, data_t, radii)
    d_interp = np.interp(hours, data_t, depths)

    fig, ax = plt.subplots(figsize=(10, 9))
    boundary_gdf.boundary.plot(ax=ax, linewidth=1, color="black")

    minx, miny, maxx, maxy = boundary_gdf.total_bounds
    margin = 5000
    ax.set_xlim(minx - margin, maxx + margin)
    ax.set_ylim(miny - margin, maxy + margin)
    ax.set_aspect("equal")
    ax.set_title("Flood Data Generator — Radial Expansion", fontsize=13)
    ax.set_xlabel("X  (m, EPSG:32618)")
    ax.set_ylabel("Y  (m, EPSG:32618)")

    origin = flood_data[0]["origin"]
    ax.plot(*origin, "r^", markersize=10, zorder=5, label="Flood origin")

    flood_fill = Circle(origin, 0, color="steelblue", alpha=0.25, zorder=2)
    flood_ring = Circle(origin, 0, fill=False, edgecolor="steelblue", linewidth=2, zorder=3)
    ax.add_patch(flood_fill)
    ax.add_patch(flood_ring)

    info = ax.text(
        0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8), zorder=6,
    )
    ax.legend(loc="lower right", fontsize=8)

    def update(frame):
        r = r_interp[frame]
        flood_fill.set_radius(r)
        flood_ring.set_radius(r)
        info.set_text(
            f"Hour        : {hours[frame]:>5.1f}\n"
            f"Flood radius: {r/1000:>6.2f} km\n"
            f"Depth origin: {d_interp[frame]:>5.2f} m"
        )
        return flood_fill, flood_ring, info

    anim = FuncAnimation(fig, update, frames=n_frames,
                         interval=interval, blit=False, repeat=repeat)
    plt.tight_layout()
    return anim


if __name__ == "__main__":
    flood_data, boundary_gdf = create_flood_data(
        boundary_path  = "jsons/Precinct_Boundaries.geojson",
        hours          = 24,
        peak_depth_m   = 1.5,
        flood_fraction = 0.9,
    )

    save_flood_data(flood_data, out_path="synthetic_data/flood_data.geojson")

    anim = animate_flood(flood_data, boundary_gdf)
    plt.show()
