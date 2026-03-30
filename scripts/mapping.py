import geopandas as gpd
import matplotlib.pyplot as plt

# Load the data
precincts = gpd.read_file('jsons/Precinct_Boundaries.geojson')
point_of_interest = gpd.read_file('jsons/random_points.geojson') # i think these are the public schools in VB
road_surfaces = gpd.read_file('jsons/Road_Surfaces.geojson')

infra_with_zones = gpd.sjoin(point_of_interest, precincts, predicate='within')
roads_with_precincts = gpd.sjoin(road_surfaces, precincts, predicate='intersects')

fig, ax = plt.subplots(figsize=(12, 12))

# 1. Draw the "Background" (All Precincts/Subcatchments)
precincts.plot(ax=ax, color='lightgrey', edgecolor='white', alpha=0.5, label='Risk Zones')

# 2. Draw the Road Surfaces
roads_with_precincts.plot(ax=ax, color='steelblue', edgecolor='navy', alpha=0.6, linewidth=0.3, label='Road Surfaces')

# 3. Draw the "Foreground" (Infrastructure Points)
# We use the joined dataframe so we can color-code them later by precinct
infra_with_zones.plot(ax=ax, marker='o', color='red', markersize=20, label='Synthetic Critical Infra')

plt.title("VB Simulator: Infrastructure mapped to Risk Zones (synthetic data)", fontsize=16, fontweight='bold')
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()
plt.show()
