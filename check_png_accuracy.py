"""
Diagnose 17 outside + fix internal noise
"""
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from shapely.ops import unary_union

# Read existing data
sampling_area = gpd.read_file("sampling_area.geojson").iloc[0].geometry
research_area = unary_union(
    list(gpd.read_file("Siming_district.geojson").to_crs("EPSG:4326").geometry) +
    list(gpd.read_file("Huli_district.geojson").to_crs("EPSG:4326").geometry)
)
luckin = pd.read_csv("xiamen_luckin_poi_wgs84.csv")
island = luckin[luckin['adname'].isin(['思明区', '湖里区'])].copy()

print("=== Distance distribution of 17 outside points ===")
distances = []
for _, row in island.iterrows():
    pt = Point(row['lon_wgs84'], row['lat_wgs84'])
    if not sampling_area.contains(pt):
        d_meters = sampling_area.distance(pt) * 111000
        distances.append((row['name'], d_meters))

# Sort by distance and see distribution
distances.sort(key=lambda x: x[1])
for name, d in distances:
    flag = "edge noise" if d < 50 else "Really outside"
    print(f"  {flag}  {d:6.1f}m  {name}")


# Repair: Morphological closing operation (first expansion and then contraction, filling internal small gaps)

print("\n=== Morphological closing operation ===")
buffer_deg = 0.00027  # ~30m

sampling_area_closed = sampling_area.buffer(buffer_deg).buffer(-buffer_deg)

# After the closed operation, it may slightly exceed the boundary of the study area (the coastline expands), and then cut again.
sampling_area_clean = sampling_area_closed.intersection(research_area)

# Save new version
gpd.GeoDataFrame(
    {"name": ["sampling_area_v2"]},
    geometry=[sampling_area_clean],
    crs="EPSG:4326"
).to_file("sampling_area_v2.geojson", driver="GeoJSON")

print(f"✓ saved sampling_area_v2.geojson")


# Recheck: How many are included now?
inside_v2, outside_v2 = 0, []
for _, row in island.iterrows():
    pt = Point(row['lon_wgs84'], row['lat_wgs84'])
    if sampling_area_clean.contains(pt):
        inside_v2 += 1
    else:
        d = sampling_area_clean.distance(pt) * 111000
        outside_v2.append((row['name'], d))

print(f"\n=== After repair ===")
print(f"Inside sampling area: {inside_v2} / {len(island)}  ({100*inside_v2/len(island):.1f}%)")
print(f"Still outside: {len(outside_v2)} 个：")
for name, d in sorted(outside_v2, key=lambda x: x[1]):
    print(f"  {d:6.1f}m  {name}")