import geopandas as gpd
import numpy as np
from shapely.geometry import LineString

SUBSTATION_FILE = "synthetic_data/substation.geojson"
OUTPUT_FILE = "synthetic_data/powerline.geojson"

# each substation connects to its 2 nearest neighbors
K = 2

if __name__ == "__main__":
    subs = gpd.read_file(SUBSTATION_FILE)

    if subs.crs is None:
        subs = subs.set_crs(epsg=4326)

    # work in meters
    subs_proj = subs.to_crs(epsg=32618).copy()

    subs_proj["x"] = subs_proj.geometry.x
    subs_proj["y"] = subs_proj.geometry.y

    edges = set()
    rows = []

    # connect each substation to its K nearest neighbors
    for i, row_i in subs_proj.iterrows():
        sid_i = row_i["substation_id"]
        p_i = row_i.geometry

        distances = []
        for j, row_j in subs_proj.iterrows():
            if i == j:
                continue
            sid_j = row_j["substation_id"]
            p_j = row_j.geometry
            d = p_i.distance(p_j)
            distances.append((sid_j, p_j, d))

        distances.sort(key=lambda x: x[2])

        for sid_j, p_j, d in distances[:K]:
            edge = tuple(sorted([sid_i, sid_j]))
            if edge not in edges:
                edges.add(edge)

                line = LineString([p_i, p_j])
                midpoint = line.interpolate(0.5, normalized=True)

                rows.append({
                    "line_id": f"L_{len(rows)+1:03d}",
                    "from_node": edge[0],
                    "to_node": edge[1],
                    "length_m": float(d),
                    "mid_x": float(midpoint.x),
                    "mid_y": float(midpoint.y),
                    "geometry": line
                })

    powerlines = gpd.GeoDataFrame(rows, crs="EPSG:32618")

    # save in lon/lat for viewing
    powerlines_out = powerlines.to_crs(epsg=4326)
    powerlines_out.to_file(OUTPUT_FILE, driver="GeoJSON")

    print(f"Saved {len(powerlines_out)} power lines to {OUTPUT_FILE}")
    print(powerlines_out[["line_id", "from_node", "to_node", "length_m"]].head())