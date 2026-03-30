import geopandas as gpd
import numpy as np

ROAD_FILE = "jsons/Road_Surfaces.geojson"
OUTPUT_FILE = "synthetic_data/roads_with_trees.geojson"

RANDOM_SEED = 42
BASE_MAX_TREE_COUNT = 8
ZERO_PROB = 0.30

rng = np.random.default_rng(RANDOM_SEED)

if __name__ == "__main__":
    roads = gpd.read_file(ROAD_FILE)

    if roads.crs is None:
        roads = roads.set_crs(epsg=4326)

    # project to meters so length is meaningful
    roads_proj = roads.to_crs(epsg=32618).copy()

    if "road_id" not in roads_proj.columns:
        roads_proj["road_id"] = [f"R_{i+1:04d}" for i in range(len(roads_proj))]

    roads_proj["length_m"] = roads_proj.geometry.length

    tree_counts = []
    heights = []

    for length_m in roads_proj["length_m"]:
        if rng.random() < ZERO_PROB:
            n = 0
        else:
            # longer roads can get slightly higher counts
            scale = min(2.0, max(1.0, length_m / 300.0))
            max_count = int(np.ceil(BASE_MAX_TREE_COUNT * scale))
            n = int(rng.integers(1, max_count + 1))

        tree_counts.append(n)
        heights.append(0.0 if n == 0 else float(rng.uniform(5.0, 15.0)))

    roads_proj["tree_count"] = tree_counts
    roads_proj["avg_tree_height"] = heights

    # save back in lon/lat for normal viewing
    roads_out = roads_proj.to_crs(epsg=4326)
    roads_out.to_file(OUTPUT_FILE, driver="GeoJSON")

    print(f"Saved {len(roads_out)} roads to {OUTPUT_FILE}")
    print(roads_out[["road_id", "tree_count", "avg_tree_height"]].head())