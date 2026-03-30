import geopandas as gpd
import numpy as np
from shapely.geometry import Point

def get_point_of_interest(polygon):
    minx, miny, maxx, maxy = polygon.bounds

    while True:
        # Generate a random point within the bounding box
        point_of_interest = Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy))

        if polygon.contains(point_of_interest):
            return point_of_interest
        
if __name__ == "__main__":
    precincts = gpd.read_file('Precinct_Boundaries.geojson')

    point_of_interest = {}
    for idx, row in precincts.iterrows():
        polygon = row['geometry']
        id = row['PRECINCT_NO']
        point_of_interest[id] = get_point_of_interest(polygon)
    # Save the points of interest to a new GeoJSON file
    gdf = gpd.GeoDataFrame({'PRECINCT_NO': list(point_of_interest.keys()), 'geometry': list(point_of_interest.values())})
    gdf.to_file('random_points.geojson', driver='GeoJSON')
