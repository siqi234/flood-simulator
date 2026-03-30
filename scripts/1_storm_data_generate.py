import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from shapely.geometry import Point
from shapely.ops import nearest_points


def create_windstorm_data(boundary_json_path, hours=24, buffer_m=500):
    gdf = gpd.read_file(boundary_json_path)

    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)

    gdf_proj = gdf.to_crs(epsg=32618)
    boundary = gdf_proj.union_all()
    minx, miny, maxx, maxy = boundary.bounds


    # target corners of the bbox
    target_start = Point(maxx, maxy)   # top-right
    target_end   = Point(minx, miny)   # bottom-left

    # closest actual points on the boundary toward those corners
    start_on_edge = nearest_points(target_start, boundary)[1]
    end_on_edge   = nearest_points(target_end, boundary)[1]

    offset_m = 1500   # try 300–800

    # move just a little outside
    start_x = start_on_edge.x + offset_m
    start_y = start_on_edge.y + offset_m
    end_x   = end_on_edge.x - offset_m
    end_y   = end_on_edge.y - offset_m

    xs = np.linspace(start_x, end_x, hours)
    ys = np.linspace(start_y, end_y, hours)

    phases = np.linspace(0, 1, hours)
    radii = 10000 + 5000 * np.sin(np.pi * phases)   # smaller radius
    winds = 30 + 30 * np.sin(np.pi * phases)

    storm_data = []
    for t in range(hours):
        storm_data.append({
            "t": t,
            "center": (float(xs[t]), float(ys[t])),
            "radius": float(radii[t]),
            "wind_speed": float(winds[t])
        })

    return storm_data, gdf_proj


def animate_windstorm(boundary_gdf, storm_data, interval=500, save_path=None, repeat=True):
    fig, ax = plt.subplots(figsize=(8, 8))

    # static boundary
    boundary_gdf.boundary.plot(ax=ax, linewidth=1)

    xs = [s["center"][0] for s in storm_data]
    ys = [s["center"][1] for s in storm_data]

    # axis limits with margin
    margin = 15000
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)
    ax.set_aspect("equal")

    ax.set_title("24-Hour Windstorm Animation")
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")

    # animated artists
    path_line, = ax.plot([], [], linewidth=2, label="Storm path")
    center_dot, = ax.plot([], [], marker="o", linestyle="None", markersize=8, label="Storm center")
    time_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")

    # start/end markers
    ax.plot(xs[0], ys[0], marker="o", markersize=8, linestyle="None", label="Start")
    ax.plot(xs[-1], ys[-1], marker="x", markersize=8, linestyle="None", label="End")

    # storm circle
    first = storm_data[0]
    storm_circle = Circle(first["center"], first["radius"], fill=False, linestyle="--", alpha=0.7)
    ax.add_patch(storm_circle)

    ax.legend()

    def init():
        path_line.set_data([], [])
        center_dot.set_data([], [])
        storm_circle.center = storm_data[0]["center"]
        storm_circle.set_radius(storm_data[0]["radius"])
        time_text.set_text("")
        return path_line, center_dot, storm_circle, time_text

    def update(frame):
        x_now = xs[:frame + 1]
        y_now = ys[:frame + 1]

        path_line.set_data(x_now, y_now)
        center_dot.set_data([xs[frame]], [ys[frame]])

        storm_circle.center = storm_data[frame]["center"]
        storm_circle.set_radius(storm_data[frame]["radius"])

        time_text.set_text(
            f"Hour: {storm_data[frame]['t']}\n"
            f"Wind: {storm_data[frame]['wind_speed']:.1f} m/s\n"
            f"Radius: {storm_data[frame]['radius']:.0f} m"
        )

        return path_line, center_dot, storm_circle, time_text

    anim = FuncAnimation(
        fig,
        update,
        frames=len(storm_data),
        init_func=init,
        interval=interval,
        blit=False,
        repeat=repeat
    )

    if save_path is not None:
        if save_path.endswith(".gif"):
            anim.save(save_path, writer="pillow")
        elif save_path.endswith(".mp4"):
            anim.save(save_path, writer="ffmpeg")
        else:
            raise ValueError("save_path must end with .gif or .mp4")

    plt.show()
    return anim


# example
storm_data, boundary_gdf = create_windstorm_data(
    "jsons/Precinct_Boundaries.geojson",
    hours=24,
    buffer_m=5000
)

# save the windstorm data as geojson file for use in other algorithms
from shapely.geometry import Point
import json

features = []
for s in storm_data:
    features.append({
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [s["center"][0], s["center"][1]]
        },
        "properties": {
            "t": s["t"],
            "radius": s["radius"],
            "wind_speed": s["wind_speed"]
        }
    })

geojson = {"type": "FeatureCollection", "features": features}
with open("synthetic_data/storm_data.geojson", "w") as f:
    json.dump(geojson, f, indent=2)

# anim = animate_windstorm(boundary_gdf, storm_data, interval=600, repeat=False)