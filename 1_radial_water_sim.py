import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon, Circle
from matplotlib.path import Path

from scripts.single_node_infra_sim import FloodSimEnv

NUM_OF_INTEREST = 108

def extract_polygons(json_data: gpd.GeoDataFrame):
    '''
        Extracts polygons from a GeoJSON file and returns a dictionary mapping precinct numbers to their corresponding geometries.
        Inputs:
            json_data: a GeoDataFrame containing the precinct boundaries, with columns 'PRECINCT_NO' and 'geometry'.
        Returns:
            poly: A dictionary where keys are precinct numbers and values are the corresponding geometries (polygons).
    '''

    areas = {json_data['PRECINCT_NO'][i]: json_data['geometry'][i] for i in range(len(json_data))} # areas = {'01': <POLYGON ((x1, y1), (x2, y2), ...)>, ...}
    # areas = dict(zip(json_data['PRECINCT_NO'], json_data['geometry']))
    polygons =  dict(sorted(areas.items()))

    return polygons

def extract_points(json_data: gpd.GeoDataFrame):
    '''
        Extracts points from a GeoJSON file and returns a dictionary mapping precinct numbers to their corresponding point geometries.
        Inputs:
            json_data: a GeoDataFrame containing the points of interest, with columns 'PRECINCT_NO' and 'geometry'.
        Returns:
            points_of_interest: A dictionary where keys are precinct numbers and values are the corresponding point geometries.
    '''
    points = {json_data['PRECINCT_NO'][i]: json_data['geometry'][i] for i in range(len(json_data))} # points_of_interest = {'01': <POINT (x, y)>, ...}
    points_of_interest = dict(sorted(points.items()))
    
    return points_of_interest


def plot_map(polygons: dict, points: dict, flood_x: float = None):
    '''
        Plots the polygons on a map using Matplotlib.

        Inputs:
            polygons: A dictionary where keys are precinct numbers and values are the corresponding geometries (polygons).
            points: A dictionary where keys are precinct numbers and values are the corresponding point geometries.
        Outputs:
            A Matplotlib plot displaying the precinct polygons and the points of interest.
        Returns:
            None
    '''
    fig, ax = plt.subplots(figsize=(12, 12))

    for precinct_no, poly in polygons.items():
        if poly.geom_type == 'Polygon':
            x, y = poly.exterior.xy
            ax.plot(x, y, color='grey', alpha=0.5)
            
            point_x = points[precinct_no].x
            point_y = points[precinct_no].y

            # facility point color changes if flood reaches it
            if flood_x is not None and point_x >= flood_x:
                point_color = 'red'
            else:
                point_color = 'green'

            ax.plot(point_x, point_y, marker='o', color=point_color, markersize=10)

            # Add text at centroid
            centroid = poly.centroid
            ax.text(centroid.x, centroid.y, str(precinct_no), fontsize=8, ha='center', va='center', fontweight='bold')

    # --- set limits first ---
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    # --- draw flood area + flood line ---
    if flood_x is not None:
        ax.axvspan(flood_x, xmax, color='lightblue', alpha=0.3)
        ax.axvline(x=flood_x, color='blue', linewidth=2)

    plt.title("VB Simulator: areas of interest", fontsize=20,fontweight='bold')
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()

def update(frame, polygons, points, ax, xmin, xmax, ymin, ymax, flood_frames, post_flood_frames, envs, activated, flood_origin, max_radius):
    '''
        Update function for the animation. This function will be called for each frame of the animation to update the plot with the current state of the polygons and points.

        Inputs:
            frame: The current frame number (not used in this implementation, but can be used to control the animation).
            polygons: A dictionary where keys are precinct numbers and values are the corresponding geometries (polygons).
            points: A dictionary where keys are precinct numbers and values are the corresponding point geometries.
            ax: The Matplotlib Axes object to update.
        returns:
            None
    '''
    ax.clear()

    if frame < flood_frames:
        progress = frame / (flood_frames - 1)
        smooth_progress = 0.5 - 0.5 * np.cos(np.pi * progress)
        flood_radius = max_radius * smooth_progress
    else:
        flood_radius = max_radius

    flood_fill = Circle(flood_origin, flood_radius, color='lightblue', alpha=0.3)
    flood_ring = Circle(flood_origin, flood_radius, color='blue', fill=False, linewidth=2)
    ax.add_patch(flood_fill)
    ax.add_patch(flood_ring)

    for precinct_no, poly in polygons.items():
        if poly.geom_type == 'Polygon':
            x, y = poly.exterior.xy
            ax.plot(x, y, color='grey', alpha=0.5)

            point_x = points[precinct_no].x
            point_y = points[precinct_no].y

            # 1. flood reaches this point -> activate its local env
            dist = np.sqrt((point_x - flood_origin[0])**2 + (point_y - flood_origin[1])**2)
            if (not activated[precinct_no]) and (dist <= flood_radius):
                activated[precinct_no] = True

            # 2. once activated, keep stepping that node's env
            env = envs[precinct_no]
            if activated[precinct_no] and sum(env.infra_states) > 0:
                state, reward, terminated, truncated, info = env.step(0)
            else:
                state = np.array([env.time, env.water_level] + env.infra_states, dtype=np.float32)

            infra_states = state[2:]

            # 3. decide point color
            if np.sum(infra_states) == 0:
                point_color = 'red'      # all 3 infrastructures failed
            elif activated[precinct_no]:
                point_color = 'orange'   # flood reached, failure simulation started
            else:
                point_color = 'green'    # not reached yet

            ax.plot(point_x, point_y, marker='o', color=point_color, markersize=10)

            # optional: show fail count
            fail_count = 3 - int(np.sum(infra_states))
            ax.text(point_x, point_y + 0.002, f"{fail_count}/3", fontsize=7, ha='center')

            centroid = poly.centroid
            ax.text(
                centroid.x, centroid.y, str(precinct_no),
                fontsize=8, ha='center', va='center', fontweight='bold'
            )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_title(f"VB Simulator: areas of interest (timestep {frame+1})", fontsize=16, fontweight='bold')
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")


if __name__ == "__main__":
    # 1 extract polygons from geojson file
    districts_file = gpd.read_file('jsons/Precinct_Boundaries.geojson')
    areas = extract_polygons(districts_file)

    points_file = gpd.read_file('jsons/random_points.geojson')
    points_of_interest = extract_points(points_file)

        # only need 10 for testing
    polygons = {k: areas[k] for k in list(areas.keys())[:NUM_OF_INTEREST]} # remove this line in the future for all areas animation
    points = {k: points_of_interest[k] for k in list(points_of_interest.keys())[:NUM_OF_INTEREST]} # remove this line in the future for all points animation

    # 2 plot the polygons
    # plot_map(polygons, points, flood_x=-76.00)
    envs = {}
    activated = {}

    for precinct_no in polygons.keys():
        env = FloodSimEnv()
        env.reset()
        envs[precinct_no] = env
        activated[precinct_no] = False

    fig, ax = plt.subplots(figsize=(12, 12))
    print("Starting ...")

    all_x = []
    all_y = []

    for poly in polygons.values():
        if poly.geom_type == 'Polygon':
            x, y = poly.exterior.xy
            all_x.extend(x)
            all_y.extend(y)

    xmin, xmax = min(all_x), max(all_x)
    ymin, ymax = min(all_y), max(all_y)

    flood_frames = 100
    post_flood_frames = 35
    n_frames = flood_frames + post_flood_frames

    flood_origin = (xmax, (ymin + ymax) / 2)
    max_radius = np.sqrt((xmax - xmin)**2 + (ymax - ymin)**2)

    ani = FuncAnimation(
    fig,
    update,
    frames=n_frames, # Frame count for the animation
    fargs=(polygons, points, ax, xmin, xmax, ymin, ymax, flood_frames, post_flood_frames, envs, activated, flood_origin, max_radius),
    interval=100, # Delay between frames in milliseconds
    repeat=False
)
    plt.show()





  

