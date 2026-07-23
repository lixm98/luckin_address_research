"""
Final fix: Patch WorldCover's "obsolete reclamation area" with case buffer
Input: sampling_area_v2.geojson (after closed operation) + luckin WGS-84 data
Output: sampling_area_final.geojson (for next step sampling)
"""
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from shapely.ops import unary_union

# Read v2 after closing operation
sampling_v2 = gpd.read_file("geojson/sampling_area_v2.geojson").iloc[0].geometry

# Reading cases in the island
luckin = pd.read_csv("csv/xiamen_luckin_poi_wgs84.csv")
island = luckin[luckin['adname'].isin(['思明区', '湖里区'])]

# Set a 50m circle for each case
# 50m Reason for selection:
#   -> Edge noise scale (30m), able to restore "really out there"
#   -< 200m control point exclusion radius, does not affect control point scattering logic
# WGS-84 down 50m ≈ 0.00045 degrees

BUFFER_DEG = 0.00045  # ~50m

case_buffers = unary_union([
    Point(row['lon_wgs84'], row['lat_wgs84']).buffer(BUFFER_DEG)
    for _, row in island.iterrows()
])

# Merge: sampling_v2 ∪ case_buffers
sampling_final = sampling_v2.union(case_buffers)

# keep
gpd.GeoDataFrame(
    {"name": ["sampling_area_final"]},
    geometry=[sampling_final],
    crs="EPSG:4326"
).to_file("geojson/sampling_area_final.geojson", driver="GeoJSON")

# Final verification: Must be 176/176 all inside
inside, outside = 0, []
for _, row in island.iterrows():
    pt = Point(row['lon_wgs84'], row['lat_wgs84'])
    if sampling_final.contains(pt):
        inside += 1
    else:
        d = sampling_final.distance(pt) * 111000
        outside.append((row['name'], d))

print(f"Final result: {inside} / {len(island)} cases inside the sampling area")
if outside:
    print("⚠️ Still outside (theoretically should be 0):")
    for name, d in outside:
        print(f"  {d:.1f}m  {name}")
else:
    print("✅ All cases have been included")