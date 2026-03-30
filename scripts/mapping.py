import geopandas as gpd
import matplotlib.pyplot as plt

# Load the data
precincts = gpd.read_file('jsons/Precinct_Boundaries.geojson')
# point_of_interest = gpd.read_file('jsons/random_points.geojson') # i think these are the public schools in VB
substation_points = gpd.read_file('synthetic_data/substation.geojson')
tower_points = gpd.read_file('synthetic_data/tower.geojson')
powerline = gpd.read_file('synthetic_data/powerline.geojson')
# road_surfaces = gpd.read_file('jsons/Road_Surfaces.geojson')
street = gpd.read_file('synthetic_data/roads_with_trees.geojson')

infra_with_zones = gpd.sjoin(substation_points, precincts, predicate='within')
# roads_with_precincts = gpd.sjoin(road_surfaces, precincts, predicate='intersects')
street_with_precincts = gpd.sjoin(street, precincts, predicate='intersects')

fig, ax = plt.subplots(figsize=(12, 12))

# 1. Draw the "Background" (All Precincts/Subcatchments)
precincts.plot(ax=ax, color='lightgrey', edgecolor='white', alpha=0.5, label='Risk Zones')

# 2. Draw the Road Surfaces
# roads_with_precincts.plot(ax=ax, color='steelblue', edgecolor='navy', alpha=0.6, linewidth=0.3, label='Road Surfaces')

# Draw the street centerlines
street_with_precincts.plot(ax=ax, color='black', linewidth=1, label='Street Centerlines')

# Draw the tower points
tower_points.plot(ax=ax, marker='^', color='orange', markersize=20, label='Synthetic Towers')

# Draw the powerlines
powerline.plot(ax=ax, color='green', linewidth=1, label='Synthetic Powerlines')

# 3. Draw the "Foreground" (Infrastructure Points)
# We use the joined dataframe so we can color-code them later by precinct
infra_with_zones.plot(ax=ax, marker='o', color='red', markersize=40, label='Synthetic Critical Infra')

plt.title("VB Simulator: Infrastructure mapped to Risk Zones (synthetic data)", fontsize=16, fontweight='bold')
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()
plt.show()
