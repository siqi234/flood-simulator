import geopandas as gpd
import numpy as np
from shapely.geometry import Point

SUBSTATION_FILE = "synthetic_data/substation.geojson"
POI_FILE = "jsons/Points_Of_Interest.geojson"
OUTPUT_FILE = "synthetic_data/tower.geojson"

RANDOM_SEED = 42
BACKUP_HOURS = 3
MAX_ASSIGN_DIST_M = 2000
MIN_FALLBACK_DIST_M = 300
MAX_FALLBACK_DIST_M = 1200

ALLOWED_TYPECODES = {
    "Police Station",
    "Fire and EMS",
    "City Offices",
    "City Operations and Maintenance",
    "City Facility - Other",
    "Public Safety",
    "EMS",
    "Fire Station",
}

rng = np.random.default_rng(RANDOM_SEED)

def random_point_near(center_point, min_dist=300, max_dist=1200):
    angle = rng.uniform(0, 2 * np.pi)
    dist = rng.uniform(min_dist, max_dist)
    x = center_point.x + dist * np.cos(angle)
    y = center_point.y + dist * np.sin(angle)
    return Point(x, y)

if __name__ == "__main__":
    subs = gpd.read_file(SUBSTATION_FILE)
    pois = gpd.read_file(POI_FILE)

    if subs.crs is None:
        subs = subs.set_crs(epsg=4326)
    if pois.crs is None:
        pois = pois.set_crs(epsg=4326)

    # project both to meters for distance checks
    subs = subs.to_crs(epsg=32618)
    pois = pois.to_crs(epsg=32618)

    pois = pois[pois["typecode"].isin(ALLOWED_TYPECODES)].copy()

    used_poi_ids = set()
    tower_rows = []

    for _, sub in subs.iterrows():
        substation_id = sub["substation_id"]
        precinct_no = sub["PRECINCT_NO"]
        sub_geom = sub.geometry

        n_towers = int(rng.choice([1, 2]))
        selected_count = 0

        if len(pois) > 0:
            candidates = pois.copy()
            candidates["dist_m"] = candidates.geometry.distance(sub_geom)
            candidates = candidates[
                (candidates["dist_m"] <= MAX_ASSIGN_DIST_M) &
                (~candidates["OBJECTID"].isin(used_poi_ids))
            ].copy()

            if len(candidates) > 0:
                n_select = min(n_towers, len(candidates))
                sampled = candidates.sample(
                    n=n_select,
                    random_state=int(rng.integers(0, 1_000_000))
                )

                for _, row in sampled.iterrows():
                    used_poi_ids.add(row["OBJECTID"])

                    tower_rows.append({
                        "tower_id": f"T_{len(tower_rows)+1:03d}",
                        "substation_id": substation_id,
                        "PRECINCT_NO": precinct_no,
                        "backup_hours": BACKUP_HOURS,
                        "source_type": "poi",
                        "poi_name": row.get("name"),
                        "poi_type": row.get("typecode"),
                        "geometry": row.geometry,
                    })
                    selected_count += 1

        while selected_count < n_towers:
            fallback_point = random_point_near(
                sub_geom,
                min_dist=MIN_FALLBACK_DIST_M,
                max_dist=MAX_FALLBACK_DIST_M
            )

            tower_rows.append({
                "tower_id": f"T_{len(tower_rows)+1:03d}",
                "substation_id": substation_id,
                "PRECINCT_NO": precinct_no,
                "backup_hours": BACKUP_HOURS,
                "source_type": "synthetic_fallback",
                "poi_name": None,
                "poi_type": None,
                "geometry": fallback_point,
            })
            selected_count += 1

    towers = gpd.GeoDataFrame(tower_rows, crs="EPSG:32618")
    towers_out = towers.to_crs(epsg=4326)
    towers_out.to_file(OUTPUT_FILE, driver="GeoJSON")

    print(f"Saved {len(towers_out)} towers to {OUTPUT_FILE}")
    print(towers_out[["tower_id", "substation_id", "PRECINCT_NO", "backup_hours", "source_type"]].head())

    print("\nFirst 5 tower coordinates:")
    for geom in towers_out.geometry.head():
        print((geom.x, geom.y))