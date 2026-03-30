import geopandas as gpd
import numpy as np
from shapely.geometry import Point

def get_point_of_interest(polygon, inward_buffer=200):
    inner_polygon = polygon.buffer(-inward_buffer)

    if inner_polygon.is_empty:
        inner_polygon = polygon

    minx, miny, maxx, maxy = inner_polygon.bounds

    while True:
        point = Point(
            np.random.uniform(minx, maxx),
            np.random.uniform(miny, maxy)
        )
        if inner_polygon.contains(point):
            return point

if __name__ == "__main__":
    precincts = gpd.read_file("jsons/Precinct_Boundaries.geojson")

    if precincts.crs is None:
        precincts = precincts.set_crs(epsg=4326)

    # keep original CRS for later export
    original_crs = precincts.crs

    # project to meters for buffering
    precincts_proj = precincts.to_crs(epsg=32618)

    points_of_interest = {}
    for _, row in precincts_proj.iterrows():
        polygon = row["geometry"]
        precinct_id = row["PRECINCT_NO"]
        points_of_interest[precinct_id] = get_point_of_interest(polygon, inward_buffer=200)

    gdf = gpd.GeoDataFrame(
    {
        "substation_id": [f"S_{pid}" for pid in points_of_interest.keys()],
        "PRECINCT_NO": list(points_of_interest.keys()),
        "geometry": list(points_of_interest.values())
    },
    crs="EPSG:32618"
)

    gdf = gdf.to_crs(epsg=4326)
    gdf.to_file("synthetic_data/substation.geojson", driver="GeoJSON")