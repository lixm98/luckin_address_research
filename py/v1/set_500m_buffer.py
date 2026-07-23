"""
500m buffer feature engineering
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
from shapely.strtree import STRtree
import matplotlib.pyplot as plt

# 0. Parameters
BUFFER_M = 500

# Projected coordinate system: UTM zone 50N (EPSG:32650), covering Fujian/Taiwan
# In the metric coordinate system, buffer(500) can correctly generate a 500m circle.
# WGS-84 (EPSG:4326) is a spherical coordinate, and the buffer distance cannot be in meters.
WGS84 = "EPSG:4326"
UTM   = "EPSG:32650"


# 1. Load samples (modeling main table)
samples = pd.read_csv("csv/samples.csv")
print(f"✓ load samples: {len(samples)} rows ({samples['case'].sum()} cases + "
      f"{(samples['case']==0).sum()} controls)")

# Convert to GeoDataFrame and project to metric coordinates
samples_gdf = gpd.GeoDataFrame(
    samples,
    geometry=[Point(r['lon_wgs84'], r['lat_wgs84']) for _, r in samples.iterrows()],
    crs=WGS84
).to_crs(UTM)


# 2. Load all POI data + project to metric
def load_pois(csv_path):
    """Read POI csv and return GeoDataFrame projected to UTM"""
    df = pd.read_csv(csv_path)
    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(r['lon_wgs84'], r['lat_wgs84']) for _, r in df.iterrows()],
        crs=WGS84
    ).to_crs(UTM)
    return gdf

print("Loading POI data...")
office       = load_pois("csv/xiamen_office_wgs84.csv")
residential  = load_pois("csv/xiamen_residential_wgs84.csv")
mall         = load_pois("csv/xiamen_mall_wgs84.csv")
metro_st     = load_pois("csv/xiamen_metro_stations_wgs84.csv")
metro_exits  = load_pois("csv/xiamen_metro_exits_wgs84.csv")

# Distance pool: Use city-wide data, including outside the island. Avoid boundary effects
luckin_all     = load_pois("csv/xiamen_luckin_poi_wgs84.csv")
starbucks_all  = load_pois("csv/xiamen_starbucks_poi_wgs84.csv")

print(f"  office: {len(office)}, residential: {len(residential)}, "
      f"mall: {len(mall)}, metro_stations: {len(metro_st)}, "
      f"metro_exits: {len(metro_exits)}")
print(f"  luckin (Citywide): {len(luckin_all)}, starbucks (Citywide): {len(starbucks_all)}")


# 3. POI counting characteristics -how many are there in the 500m buffer?
# Method: Use spatial index (STRtree) to find the POI falling in each buffer
# STRtree is orders of magnitude faster than brute force comparison

def count_pois_in_buffer(sample_points_utm, poi_gdf, buffer_m):
    """
    Draw the buffer for each sample and count the POIs in it.
    sample_points_utm: GeoSeries (UTM)
    poi_gdf: GeoDataFrame (UTM)
    Return: list[int]
    """
    poi_points = list(poi_gdf.geometry)
    tree = STRtree(poi_points)

    counts = []
    for pt in sample_points_utm:
        # query uses buffer as the query window and returns the index of potential candidate POIs.
        # Then accurately judge distance <= buffer_m
        candidates = tree.query(pt.buffer(buffer_m))
        n = sum(1 for i in candidates if pt.distance(poi_points[i]) <= buffer_m)
        counts.append(n)
    return counts

print("\nCalculating POI counts in 500m buffer...")
samples_gdf['office_count']        = count_pois_in_buffer(samples_gdf.geometry, office, BUFFER_M)
samples_gdf['residential_count']   = count_pois_in_buffer(samples_gdf.geometry, residential, BUFFER_M)
samples_gdf['mall_count']          = count_pois_in_buffer(samples_gdf.geometry, mall, BUFFER_M)
samples_gdf['metro_station_count'] = count_pois_in_buffer(samples_gdf.geometry, metro_st, BUFFER_M)
print("  ✓ 4 count features calculated")


# 4. Distance feature -to the nearest point of a certain type
def dist_to_nearest(sample_points_utm, poi_gdf, exclude_self_name=None, sample_names=None):
    """
    Calculate the distance to the nearest POI for each sample (meters)
    exclude_self_name: If not None, exclude "self" by name field (used in case when calculating the nearest luckin)
    """
    poi_points = list(poi_gdf.geometry)
    poi_names = poi_gdf['name'].tolist() if 'name' in poi_gdf.columns else [None]*len(poi_gdf)
    tree = STRtree(poi_points)

    distances = []
    for i, pt in enumerate(sample_points_utm):
        nearest_idx = tree.nearest(pt)
        # If you are yourself, find the next closest one
        if exclude_self_name and sample_names is not None:
            sname = sample_names.iloc[i]
            if poi_names[nearest_idx] == sname:
                # Use a large buffer to check multiple candidates and pick the closest one other than your own.
                candidates = tree.query(pt.buffer(5000))
                candidates = [j for j in candidates if poi_names[j] != sname]
                if candidates:
                    nearest_idx = min(candidates, key=lambda j: pt.distance(poi_points[j]))
                else:
                    # Extreme case: There is no other similar type within 5km (basically this will not happen), use nan
                    distances.append(np.nan)
                    continue
        distances.append(pt.distance(poi_points[nearest_idx]))
    return distances

print("\nCalculating distance features...")
# Recent luckin: case to exclude yourself (match with the same name)
samples_gdf['dist_to_nearest_luckin'] = dist_to_nearest(
    samples_gdf.geometry, luckin_all,
    exclude_self_name=True, sample_names=samples_gdf['name']
)
samples_gdf['dist_to_nearest_starbucks']   = dist_to_nearest(samples_gdf.geometry, starbucks_all)
samples_gdf['dist_to_nearest_metro_exit']  = dist_to_nearest(samples_gdf.geometry, metro_exits)
print("  ✓ 3 distance features calculated")


# 5. Save features.csv
# Output back to WGS-84 coordinates (remain consistent with the original sample file), but retain all features
samples_gdf_wgs = samples_gdf.to_crs(WGS84)
features = pd.DataFrame({
    "lon_wgs84": samples_gdf_wgs.geometry.x,
    "lat_wgs84": samples_gdf_wgs.geometry.y,
    "case": samples_gdf['case'],
    "name": samples_gdf['name'],
    "office_count":              samples_gdf['office_count'],
    "residential_count":         samples_gdf['residential_count'],
    "mall_count":                samples_gdf['mall_count'],
    "metro_station_count":       samples_gdf['metro_station_count'],
    "dist_to_nearest_luckin":    samples_gdf['dist_to_nearest_luckin'],
    "dist_to_nearest_starbucks": samples_gdf['dist_to_nearest_starbucks'],
    "dist_to_nearest_metro_exit":samples_gdf['dist_to_nearest_metro_exit'],
})
features.to_csv("csv/features.csv", index=False, encoding="utf-8-sig")
print(f"\n✓ Features saved to csv/features.csv ({len(features)} rows)")


# 6. Descriptive statistics: case vs control comparison
print("\n=== case vs control Feature Means ===")
feature_cols = ['office_count', 'residential_count', 'mall_count', 'metro_station_count',
                'dist_to_nearest_luckin', 'dist_to_nearest_starbucks', 'dist_to_nearest_metro_exit']

summary = features.groupby('case')[feature_cols].agg(['mean', 'median'])
print(summary.round(1))

# If the count class feature mean of case > control, it means that Ruixing "prefers" this type of position (preliminary signal)
# If the distance characteristics of the case < control, it means that Ruixing is "close" to this type of target.
print("\npreliminary signal（case > control It means that Ruixing prefers positions with high characteristics):")
for col in feature_cols:
    case_mean = features[features['case']==1][col].mean()
    ctrl_mean = features[features['case']==0][col].mean()
    diff = case_mean - ctrl_mean
    direction = "case higher" if diff > 0 else "case lower"
    print(f"  {col:35s}: {direction}  (Difference {diff:+.1f})")


# 7. Visual inspection: box plot
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.flatten()

for i, col in enumerate(feature_cols):
    ax = axes[i]
    data = [features[features['case']==0][col], features[features['case']==1][col]]
    bp = ax.boxplot(data, labels=['Control', 'Case'], patch_artist=True, showfliers=False)
    bp['boxes'][0].set_facecolor('steelblue')
    bp['boxes'][1].set_facecolor('crimson')
    ax.set_title(col, fontsize=10)
    ax.grid(axis='y', alpha=0.3)

axes[-1].axis('off')  # Leave the last subplot blank
plt.suptitle("Feature distribution: cases (red) vs controls (blue)", fontsize=12)
plt.tight_layout()
plt.savefig("png/feature_check.png", dpi=130, bbox_inches='tight')
plt.close()
print(f"\n✓ Check plot saved to png/feature_check.png")