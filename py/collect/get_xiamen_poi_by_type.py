"""
Capture POIs in Xiamen Island by typecode + rectangular grid slice.

Usage:
    python get_xiamen_poi_by_type.py

Modify the typecode and output file name in CONFIGS at the bottom to capture different categories.
"""

import requests
import pandas as pd
import time
from itertools import product

# 1. Basic configuration

AMAP_KEY = "e959958de5f40b3052689849233760f8"
BASE_URL = "https://restapi.amap.com/v3/place/polygon"

# Approximate latitude and longitude bounding box of Xiamen Island (Siming + Huli)
# This is a rectangle slightly larger than the island, encompassing the entire island
# Then use Siming + Huli’s precise polygons to crop unnecessary points.
ISLAND_BBOX = {
    "lon_min": 118.06,   # Western boundary (a little beyond the coastline on the west side of the island)
    "lon_max": 118.20,   # eastern border
    "lat_min": 24.43,    # Southern boundary (a little beyond the southern end of the island)
    "lat_max": 24.56,    # Northern boundary (a little outside the northern end of Huli District)
}

# Grid slice density: cut the rectangle into N×N small squares
# The denser the grid, the fewer POIs per grid, and the less likely it is to reach the top; but the more requests = N×N.
# 8×8 = 64 cells, enough for medium-density POI categories on the island
GRID_N = 8

# 20 items per page is the maximum value of Amap, and the maximum number of pages is 100, so the hard upper limit for a single search = 2000 items
PAGE_SIZE = 20
MAX_PAGE = 100

# Wait politely to avoid triggering frequency limits
SLEEP_BETWEEN_REQUESTS = 0.3


# 2. Generate mesh

def build_grid(bbox, n):
    """
    Cut the rectangle bbox into n×n small rectangles.
    Returns [(lon_min, lat_min, lon_max, lat_max), ...] total n*n tuples.

    For example, when n=2, a large square will be cut into 4 small squares:
        ┌────┬────┐
        │ 1 │ 2 │
        ├────┼────┤
        │ 3 │ 4 │
        └────┴────┘
    """
    lon_step = (bbox["lon_max"] - bbox["lon_min"]) / n
    lat_step = (bbox["lat_max"] - bbox["lat_min"]) / n

    cells = []
    for i, j in product(range(n), range(n)):
        cell_lon_min = bbox["lon_min"] + i * lon_step
        cell_lon_max = bbox["lon_min"] + (i + 1) * lon_step
        cell_lat_min = bbox["lat_min"] + j * lat_step
        cell_lat_max = bbox["lat_min"] + (j + 1) * lat_step
        cells.append((cell_lon_min, cell_lat_min, cell_lon_max, cell_lat_max))
    return cells


# 3. Capture all POIs of a grid + a typecode (turn the page until there is no data)

def fetch_one_cell(typecode, cell):
    """
    In a small rectangular cell, flip through all POIs with the specified typecode.

    Amap's polygon parameter format: upper left corner longitude and latitude; lower right corner longitude and latitude
    Note that the polygon rectangle is written as "upper left → lower right", not "lower left → upper right"!
    """
    lon_min, lat_min, lon_max, lat_max = cell

    # Upper left = (lon_min, lat_max), lower right = (lon_max, lat_min)
    polygon_str = f"{lon_min},{lat_max}|{lon_max},{lat_min}"

    pois_in_cell = []

    for page in range(1, MAX_PAGE + 1):
        params = {
            "key": AMAP_KEY,
            "types": typecode,        # Filter by typecode
            "polygon": polygon_str,   # Limit search rectangle
            "offset": PAGE_SIZE,
            "page": page,
            "extensions": "base",     # base is enough, all will be slower
            "output": "json",
        }

        try:
            r = requests.get(BASE_URL, params=params, timeout=10)
            data = r.json()
        except Exception as e:
            print(f"  [Request exception] {e}")
            break

        if data.get("status") != "1":
            print(f"  [API Report an error] {data.get('info')}")
            break

        pois = data.get("pois", [])
        if not pois:
            # There is no data on this page, which means the page has been turned over.
            break

        pois_in_cell.extend(pois)

        # If this page gets < PAGE_SIZE, it means it is the last page and stops early.
        if len(pois) < PAGE_SIZE:
            break

        time.sleep(SLEEP_BETWEEN_REQUESTS)

    # If a cell really gets 2,000 items (top warning), send a reminder
    if len(pois_in_cell) >= MAX_PAGE * PAGE_SIZE:
        print(f"  ⚠️ Warning: cell {cell} Top ({len(pois_in_cell)} strip), it is recommended to increase GRID_N and re-grab")

    return pois_in_cell


# 4. Main process: traverse all grids and capture one type of POI

def fetch_poi_by_type(typecode, output_csv, label=""):
    """
    Capture all POIs with specified typecode on the island and save them as CSV.
    """
    cells = build_grid(ISLAND_BBOX, GRID_N)
    print(f"\n=== Start crawling {label} (typecode={typecode}) ===")
    print(f"共 {len(cells)} grids")

    all_pois = []

    for idx, cell in enumerate(cells, start=1):
        cell_pois = fetch_one_cell(typecode, cell)
        print(f"  grid {idx}/{len(cells)}: get {len(cell_pois)} strip")
        all_pois.extend(cell_pois)
        time.sleep(SLEEP_BETWEEN_REQUESTS)

    print(f"Crawl completed, original total (including cross-grid repeats):{len(all_pois)}")

    if not all_pois:
        print("⚠️ Didn't catch any data, check typecode or network.")
        return

    # Organized into DataFrame
    rows = []
    for poi in all_pois:
        location = poi.get("location", "")
        lon, lat = (location.split(",") + [None, None])[:2] if location else (None, None)
        rows.append({
            "id": poi.get("id"),               # Key fields used to remove duplicates
            "name": poi.get("name"),
            "type": poi.get("type"),
            "typecode": poi.get("typecode"),
            "address": poi.get("address"),
            "adname": poi.get("adname"),
            "longitude": lon,
            "latitude": lat,
            "location": location,
        })
    df = pd.DataFrame(rows)

    # Deduplication based on the POI id of Amap -near the boundaries of adjacent grids, the same POI may be returned by both grids
    df_dedup = df.drop_duplicates(subset=["id"])
    print(f"After deduplication:{len(df_dedup)} strip")

    df_dedup.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"saved to {output_csv}")


# 5. Configure what to capture for each type of POI

CONFIGS = [
    # (typecode, output file name, Chinese label)
    ("120201", "csv/xiamen_office.csv",       "写字楼"),
    ("120300", "csv/xiamen_residential.csv",  "住宅区"),
    ("060100", "csv/xiamen_mall.csv",         "商场"),
    ("150500", "csv/xiamen_metro.csv",        "地铁站"),
    # If you still want to catch Starbucks: It is more appropriate to change BASE_URL and use keywords mode, which is not in this script.
]


if __name__ == "__main__":
    for typecode, output_csv, label in CONFIGS:
        fetch_poi_by_type(typecode, output_csv, label)
