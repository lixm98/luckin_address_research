"""
Moran's I residual spatial autocorrelation test
=========================================
Function:
  Detecting whether logistic residuals are spatially clustered
  → Decide if GWLR is needed
"""

import pandas as pd
import numpy as np
from libpysal.weights import DistanceBand, KNN
from esda.moran import Moran, Moran_Local
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point


# 1. Load residuals
df = pd.read_csv("residuals.csv")
print(f"✓ Load residuals: {len(df)} points")

# Projected to the metric coordinate system (UTM 50N) to facilitate calculation of "distance to neighbors"
gdf = gpd.GeoDataFrame(
    df,
    geometry=[Point(r['lon_wgs84'], r['lat_wgs84']) for _, r in df.iterrows()],
    crs="EPSG:4326"
).to_crs("EPSG:32650")

# Extract coordinates (meters) and residuals
coords = np.array([(p.x, p.y) for p in gdf.geometry])
residuals = df['pearson_residual'].values


# 2. Construct a spatial weight matrix -two methods for sensitivity analysis
# Method A: KNN (K=8)
#   Fixed 8 nearest neighbors for each point, regardless of actual distance
#   Advantages: Ensure that every point has neighbors
print("\n=== Construct spatial weight matrix ===")

w_knn = KNN.from_array(coords, k=8)
w_knn.transform = 'r'  # Row normalization: the sum of the weights of each row is 1 for easy interpretation
print(f"  KNN (k=8): Average neighbors = {w_knn.mean_neighbors:.1f}")

# Method B: Distance Band (1000m)
#   Taking 1000m as the radius, all points within the radius are neighbors.
#   Advantages: clear physical meaning
w_dist = DistanceBand.from_array(coords, threshold=1000, binary=True)
w_dist.transform = 'r'
print(f"  Distance (1km): Average neighbors = {w_dist.mean_neighbors:.1f}")


# 3. Overall Moran's I -overall degree of concentration
# Run two sets of weight matrices to ensure consistency of results.

print("\n=== Global Moran's I (Residual Spatial Autocorrelation) ===")

moran_knn = Moran(residuals, w_knn, permutations=999)
print(f"\nKNN (k=8):")
print(f"  Moran's I       = {moran_knn.I:.4f}")
print(f"  expect E[I]       = {moran_knn.EI:.4f}")
print(f"  z value            = {moran_knn.z_sim:.4f}")
print(f"  p value            = {moran_knn.p_sim:.4f}")

moran_dist = Moran(residuals, w_dist, permutations=999)
print(f"\nDistance (1km):")
print(f"  Moran's I       = {moran_dist.I:.4f}")
print(f"  expect E[I]       = {moran_dist.EI:.4f}")
print(f"  z value            = {moran_dist.z_sim:.4f}")
print(f"  p value            = {moran_dist.p_sim:.4f}")


# 4. Interpretation + Determine whether GWLR is needed
print("\n=== Interpretation ===")

def interpret_moran(I, p, name):
    if p < 0.05:
        if I > 0:
            verdict = "✅ Significant positive spatial autocorrelation — residuals are clustered, GWLR recommended"
        else:
            verdict = "⚠️ Significant negative spatial autocorrelation — residuals show checkerboard pattern, rare in urban studies"
    else:
        verdict = "❌ No significant spatial autocorrelation — logistic model has absorbed main spatial structure"

    print(f"  [{name}] I={I:.4f}, p={p:.4f}  →  {verdict}")

interpret_moran(moran_knn.I,  moran_knn.p_sim,  "KNN  k=8")
interpret_moran(moran_dist.I, moran_dist.p_sim, "Dist 1km")


# 5. Moran's I scatter plot -visualizing clustering patterns
# Scatter plot: x = residual, y = mean of neighbor residuals
# Four quadrants: HH = high value neighbor high value (hot spot); LL = low value neighbor low value (cold spot)
# HL/LH = outliers (high values surrounded by low values /vice versa)

from libpysal.weights import lag_spatial

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, w, label, mo in [
    (axes[0], w_knn,  "KNN (k=8)",       moran_knn),
    (axes[1], w_dist, "Distance (1km)",  moran_dist)
]:
    lag = lag_spatial(w, residuals)  # Weighted average residual of neighbors
    res_std  = (residuals - residuals.mean()) / residuals.std()
    lag_std  = (lag - lag.mean()) / lag.std()

    ax.scatter(res_std, lag_std, alpha=0.5, s=20, c='steelblue', edgecolor='none')
    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5)
    # Fitted line slope = Moran's I
    z = np.polyfit(res_std, lag_std, 1)
    ax.plot(res_std, np.poly1d(z)(res_std), 'r-', lw=1.5,
            label=f"slope = {mo.I:.3f}")
    ax.set_xlabel("Standardised residual")
    ax.set_ylabel("Spatial lag (neighbour mean)")
    ax.set_title(f"Moran's I scatter — {label}\np = {mo.p_sim:.4f}")
    ax.legend()

    # Mark the four quadrants
    ax.text(0.95, 0.95, 'HH', ha='right', va='top',  transform=ax.transAxes, fontsize=11, color='red')
    ax.text(0.05, 0.05, 'LL', ha='left',  va='bottom', transform=ax.transAxes, fontsize=11, color='blue')
    ax.text(0.05, 0.95, 'LH', ha='left',  va='top',  transform=ax.transAxes, fontsize=11, color='gray')
    ax.text(0.95, 0.05, 'HL', ha='right', va='bottom', transform=ax.transAxes, fontsize=11, color='gray')

plt.tight_layout()
plt.savefig("moran_scatter.png", dpi=150, bbox_inches='tight')
plt.close()
print("\n✓ Moran's I scatter plot saved to moran_scatter.png")


# 6. Local Moran's I (LISA) -find out the specific location of clustering
# Overall Moran's I only tells you "whether there is a crowd", LISA tells you "where the crowd is"
# Classify each point: HH (hot spot) /LL (cold spot) /HL /LH (abnormal) /ns (not significant)

lisa = Moran_Local(residuals, w_knn, permutations=999)

# Classification
q = lisa.q  # 1=HH, 2=LH, 3=LL, 4=HL
sig = lisa.p_sim < 0.05
labels = np.array(['ns'] * len(residuals), dtype=object)
labels[(sig) & (q == 1)] = 'HH'
labels[(sig) & (q == 3)] = 'LL'
labels[(sig) & (q == 2)] = 'LH'
labels[(sig) & (q == 4)] = 'HL'

print("\n=== LISA Classification (which points are clustered together)===")
from collections import Counter
counts = Counter(labels)
print(f"  HH (hot spot - high residual clustering): {counts.get('HH', 0)}  → Model underestimated Luckin density areas")
print(f"  LL (cold spot - low residual clustering): {counts.get('LL', 0)}  → Model overestimated Luckin density areas")
print(f"  HL/LH (anomaly points): {counts.get('HL', 0) + counts.get('LH', 0)}")
print(f"  ns (not significant): {counts.get('ns', 0)}")


# 7. LISA map -draw the clustered areas
df['lisa_label'] = labels

# Switch back to WGS-84 drawing
gdf_wgs = gdf.to_crs("EPSG:4326")
df['lon_plot'] = [p.x for p in gdf_wgs.geometry]
df['lat_plot'] = [p.y for p in gdf_wgs.geometry]

color_map = {'HH': 'red', 'LL': 'blue', 'HL': 'pink', 'LH': 'lightblue', 'ns': 'lightgray'}
fig, ax = plt.subplots(figsize=(11, 10))

for label, color in color_map.items():
    sub = df[df['lisa_label'] == label]
    if len(sub) == 0: continue
    size = 35 if label != 'ns' else 8
    alpha = 0.85 if label != 'ns' else 0.3
    ax.scatter(sub['lon_plot'], sub['lat_plot'], c=color, s=size, alpha=alpha,
               label=f"{label} (n={len(sub)})", edgecolor='none')

ax.set_title("LISA cluster map — residual spatial autocorrelation")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.legend(loc='upper left')
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig("lisa_map.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ LISA map saved to lisa_map.png")


# Save LISA results for subsequent GWLR use
df[['lon_wgs84', 'lat_wgs84', 'case', 'pred_proba', 'pearson_residual', 'lisa_label']].to_csv(
    "moran_lisa_results.csv", index=False, encoding='utf-8-sig'
)
print("✓ LISA results saved to moran_lisa_results.csv")