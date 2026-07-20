"""
Generate a "sampling area" polygon
=========================================
Process:
  1. Merge Siming + Huli administrative boundaries → study area
  2. Extract all "non-built-up area" pixels from WorldCover and convert them into polygons
  3. Add OSM airport polygon (WorldCover mistakenly regards the runway as a built-up area and needs to be filled manually)
  4. Exclusion area = non-built-up area ∪ Airport
  5. Sampling area = study area − exclusion area
  6. Visually superimpose case points and do rationality checks
"""

import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.features import shapes as rio_shapes
from shapely.geometry import shape, Point
from shapely.ops import unary_union
import matplotlib.pyplot as plt
import pandas as pd


# Step 1: Merge administrative boundaries → study area
# Why: There are two independent area files under DataV and need to be combined into one
# unary_union will glue two polygons together and automatically eliminate internal boundaries along "the strait opposite Gulangyu Island"
# This research_area is then both the outer boundary of the sampleable area and used to clip the WorldCover raster.

siming = gpd.read_file("Siming_district.geojson").to_crs("EPSG:4326")
huli   = gpd.read_file("Huli_district.geojson").to_crs("EPSG:4326")

research_area = unary_union(list(siming.geometry) + list(huli.geometry))
print(f"✓Step 1: The merger of the research areas is completed (Siming + Huli)")


# Step 2: Extract the "non-built-up area" polygon from WorldCover
# WorldCover pixel value: 10=Woods 20=Shrubs 30=Grassland 40=Farmland 50=Built-up area
#                   60=bare land 80=water body 90=wetland
# 50 (built-up area) is the only place where Luckin can be opened, and everything else is included in the "excluded area"
#
# Processing logic:
#   a) Clip the raster with the study area polygon (avoid processing the entire 3°×3° data)
#   b) Mark pixels "not 50 and within the study area"
#   c) Convert these pixels from raster → vector (polygon)
#   d) Merge all polygons into one large multipolygon

with rasterio.open("xiamen_worldcover_2021.tif") as src:
    # rio_mask: clip raster with study area polygon
    # crop=True will directly cut off the pixels outside the range, and the resulting raster will only cover the vicinity of the study area.
    # nodata=0 means pixels outside the study area are marked as 0
    out_image, out_transform = rio_mask(
        src, [research_area], crop=True, nodata=0
    )
    raster_data = out_image[0]  # WorldCover has only one band

print(f"  Grid size after cropping: {raster_data.shape}")

# Construct a Boolean mask: True = this pixel is "unbuilt up" and is within the study area
# (raster_data != 50) → Not a built-up area
# (raster_data != 0) → not nodata (i.e. falls within the study area)
non_built_up_mask = (raster_data != 50) & (raster_data != 0)
print(f"  Non-built-up area pixels: {non_built_up_mask.sum():,}")

# Raster → Vector: Aggregate consecutive True pixels into polygons
# The mask parameter allows rio_shapes to only work in the True area to avoid traversing the entire image.
polygons = []
for geom_dict, val in rio_shapes(
    non_built_up_mask.astype('uint8'),
    mask=non_built_up_mask,
    transform=out_transform   # Use clipped affine transformation to ensure correct coordinates
):
    polygons.append(shape(geom_dict))

print(f"  Raster to Vector: {len(polygons)} independent polygons")

# Merge hundreds or thousands of small polygons into one large one (adjacent ones will be automatically merged)
non_built_up = unary_union(polygons)
print(f"✓ Step 2: Non-built-up area extraction completed")


# Step 3: Load the airport polygon
# The airport runway is wrongly divided into 50 (built-up area) in WorldCover and needs to be supplemented separately from OSM.
# Combine all airport elements (runways, aprons, terminals) into one polygon

airport = gpd.read_file("xiamen_airport.geojson").to_crs("EPSG:4326")
# Only keep polygon and filter out possible LineString /Point
airport = airport[airport.geometry.type.isin(['Polygon', 'MultiPolygon'])]
airport_polygon = unary_union(list(airport.geometry))
print(f"✓ Step 3: Airport polygon loaded ( {len(airport)} features )")


# Step 4: Synthesize exclusion zones
# Excluded Area = Unbuilt Area ∪ Airport
# Combine two independent polygons into one to facilitate the "digging" operation in the next step

exclusion_area = unary_union([non_built_up, airport_polygon])
print(f"✓ Step 4: Exclusion zone synthesis completed")


# Step 5: Sampling area = study area − exclusion area
# difference(): "dig out" the exclusion area from the study area, resulting in a polygon with "holes"
# These holes = Yuandang Lake, Wuyuan Bay, airport, various parks and mountains, etc.
# The final sampleable area is the legal scattering range of the control points.

sampling_area = research_area.difference(exclusion_area)

# Convert it to GeoDataFrame and save it as GeoJSON to facilitate subsequent sampling and dissertation plot reuse.
sampling_gdf = gpd.GeoDataFrame(
    {"name": ["sampling_area"]},
    geometry=[sampling_area],
    crs="EPSG:4326"
)
sampling_gdf.to_file("sampling_area.geojson", driver="GeoJSON")
print(f"✓ Step 5: The sampleable area has been saved to sampling_area.geojson")


# Step 6: Visual inspection
# Stack 4 layers:
#   Bottom layer: Study area border (light gray)
#   Middle level: exclusion zone (dark grey, marks water/park/airport)
#   Middle layer: sampleable area (light green filling)
#   Top layer: 176 Luckin case points on the island (red)
# Check the target: the red dots must fall almost entirely in the green area

luckin = pd.read_csv("xiamen_luckin_poi_wgs84.csv")
island_luckin = luckin[luckin['adname'].isin(['思明区', '湖里区'])].copy()

fig, ax = plt.subplots(figsize=(12, 10))

# Layer 1: Study area outer frame (lowest layer, used as overall reference)
gpd.GeoSeries([research_area], crs="EPSG:4326").plot(
    ax=ax, facecolor='lightgray', edgecolor='black', linewidth=1, alpha=0.3
)
# Layer 2: Excluded area (dark gray fill, water and airport outline visible)
gpd.GeoSeries([exclusion_area], crs="EPSG:4326").plot(
    ax=ax, facecolor='dimgray', edgecolor='none', alpha=0.6
)
# Layer 3: Sampling area (green filling, legal position of control points)
gpd.GeoSeries([sampling_area], crs="EPSG:4326").plot(
    ax=ax, facecolor='lightgreen', edgecolor='none', alpha=0.6
)
# Layer 4: case points (topmost)
ax.scatter(
    island_luckin['lon_wgs84'], island_luckin['lat_wgs84'],
    c='red', s=18, alpha=0.85, edgecolor='darkred', linewidth=0.3,
    label=f'Luckin cases (n={len(island_luckin)})'
)

ax.set_title("Sanity check: cases should fall inside the green sampling area")
ax.set_xlabel("Longitude (WGS-84)")
ax.set_ylabel("Latitude (WGS-84)")
ax.legend(loc='upper left')
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig("sampling_area_check.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Step 6: The check image has been saved to sampling_area_check.png")


# Step 7: Numerical check -how many cases are left out?
# Not only need to look at the picture, but also use contains() to accurately judge
# Most cases should be included; a few cases close to the coastline may fall out due to a few meters difference in boundary accuracy.

inside, outside = 0, []
for _, row in island_luckin.iterrows():
    pt = Point(row['lon_wgs84'], row['lat_wgs84'])
    if sampling_area.contains(pt):
        inside += 1
    else:
        outside.append((row['name'], row['lon_wgs84'], row['lat_wgs84']))

print(f"\n=== Numerical Check ===")
print(f"Inside the sampleable area: {inside} / {len(island_luckin)}  "
      f"({100*inside/len(island_luckin):.1f}%)")
print(f"Outside: {len(outside)}")
if outside:
    print("\nCases outside (maximum 10): ")
    for name, lon, lat in outside[:10]:
        print(f"  {name}  ({lon:.5f}, {lat:.5f})")
    print("\nIf all are near coastlines/boundaries → accuracy issue, acceptable")
    print("If any clearly fall into parks/airports → need debug")