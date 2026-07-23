"""
Spread 880 control points (1:5 case-control ratio)
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
from shapely.strtree import STRtree
import matplotlib.pyplot as plt
import math


# 0. Parameter + random seed
# random_state=42 is guaranteed to be reproducible -this number should be written in dissertation methodology

RANDOM_SEED = 42
N_CASES = 176              # Lucky Number on the Island
CASE_CONTROL_RATIO = 5     # 1:5 (main model)
N_CONTROLS = N_CASES * CASE_CONTROL_RATIO  # 880

MIN_DIST_CASE_M = 200     # case-control minimum distance
MIN_DIST_CTRL_M = 100     # control-control minimum distance

# Convert meters to "degrees": Xiamen is approximately 24.5°N, and each degree of latitude ≈ 111000m
# The longitude direction is because the curvature of the earth is multiplied by cos (latitude), but the small-scale difference of 200m can be ignored
# Here we use simplified conversion, which is statistically accurate enough.
MIN_DIST_CASE_DEG = MIN_DIST_CASE_M / 111000
MIN_DIST_CTRL_DEG = MIN_DIST_CTRL_M / 111000

np.random.seed(RANDOM_SEED)


# 1. Load data
# sampling_area_final -the "sampling area" polygon generated in the previous step
# luckin (176 on the island) —— case point

sampling = gpd.read_file("geojson/sampling_area_final.geojson").iloc[0].geometry

luckin = pd.read_csv("csv/xiamen_luckin_poi_wgs84.csv")
cases = luckin[luckin['adname'].isin(['思明区', '湖里区'])].copy().reset_index(drop=True)
case_points = [Point(r['lon_wgs84'], r['lat_wgs84']) for _, r in cases.iterrows()]

print(f"✓ load: sampling_area + {len(cases)} 个 case")


# 2. Prepare spatial index (speed up distance check)
# Simple method: each time a control is generated, traverse 176 cases and calculate the distance → O(N×880) It’s okay
# But the control-control check becomes O(880²) = 770,000 distance calculations
# Using the STRtree spatial index can reduce the "finding nearest neighbor" from O(N) to O(log N), and produce results in seconds.

case_tree = STRtree(case_points)


# 3. Reject sampling
# Algorithm:
#   a) Randomly generate (lon, lat) in the enclosing rectangle of sampling_area
#   b) Check whether it falls within sampling_area (reject "holes" and overflow parts in the rectangle)
#   c) Check the distance to the nearest case >= 200m
#   d) Check the nearest distance to existing control >= 100m
#   e) All pass → accept; any one fails → discard and try again

minx, miny, maxx, maxy = sampling.bounds  # bounding rectangle

controls = []          # Store Point objects (for spatial indexing)
control_records = []   # Save dictionary (for output csv)
attempts = 0
max_attempts = 200_000  # Prevent it from getting stuck

while len(controls) < N_CONTROLS and attempts < max_attempts:
    attempts += 1

    # (a) Randomly generate candidate points in bbox
    lon = np.random.uniform(minx, maxx)
    lat = np.random.uniform(miny, maxy)
    pt = Point(lon, lat)

    # (b) Must be within sampling_area
    if not sampling.contains(pt):
        continue

    # (c) The nearest case must be >= 200m
    # STRtree.nearest returns the index of the nearest neighbor
    nearest_case_idx = case_tree.nearest(pt)
    if pt.distance(case_points[nearest_case_idx]) < MIN_DIST_CASE_DEG:
        continue

    # (d) The distance to the nearest control must be >= 100m (if there is already a control)
    if controls:
        ctrl_tree = STRtree(controls)  # Each rebuild is a bit slow, but 880 is acceptable
        nearest_ctrl_idx = ctrl_tree.nearest(pt)
        if pt.distance(controls[nearest_ctrl_idx]) < MIN_DIST_CTRL_DEG:
            continue

    # All passed, accepted
    controls.append(pt)
    control_records.append({
        "lon_wgs84": lon,
        "lat_wgs84": lat,
        "case": 0,
    })

    if len(controls) % 100 == 0:
        print(f"  Generated {len(controls)}/{N_CONTROLS}  (number of attempts {attempts})")

acceptance_rate = len(controls) / attempts * 100
print(f"\n✓ Generation completed:{len(controls)} 个 control, try {attempts} times, acceptance rate {acceptance_rate:.1f}%")

if len(controls) < N_CONTROLS:
    print(f"⚠️ Warning: Target number not reached. Possible reasons: sampling_area too small or 200m exclusion too strict")


# 4. Output controls.csv
controls_df = pd.DataFrame(control_records)
controls_df.to_csv("csv/controls.csv", index=False, encoding="utf-8-sig")
print(f"✓ Generated controls.csv ({len(controls_df)} rows)")


# 5. Merge case + control → samples.csv (modeling main table)
# This is the input for all subsequent modeling
# label：case=1, control=0

cases_out = cases[['lon_wgs84', 'lat_wgs84', 'name']].copy()
cases_out['case'] = 1

controls_out = controls_df.copy()
controls_out['name'] = '(control)'

samples = pd.concat([cases_out, controls_out], ignore_index=True)
samples = samples[['lon_wgs84', 'lat_wgs84', 'case', 'name']]
samples.to_csv("csv/samples.csv", index=False, encoding="utf-8-sig")

print(f"✓ Generated samples.csv ({len(samples)} rows = {samples['case'].sum()} cases + {(samples['case']==0).sum()} controls)")


# 6. Visual inspection
# Three-layer diagram:
#   Bottom: sampling_area (light green)
#   Center: case (red dot)
#   Top: control (blue dot, translucent)
# Check: controls should be evenly distributed in the green area and not crowded in a corner; there should be a "no-fly circle" around the red dot

fig, ax = plt.subplots(figsize=(12, 10))

gpd.GeoSeries([sampling], crs="EPSG:4326").plot(
    ax=ax, facecolor='lightgreen', edgecolor='gray', linewidth=0.5, alpha=0.5
)
ax.scatter(controls_df['lon_wgs84'], controls_df['lat_wgs84'],
           c='steelblue', s=8, alpha=0.5, label=f'Controls (n={len(controls_df)})')
ax.scatter(cases['lon_wgs84'], cases['lat_wgs84'],
           c='red', s=20, alpha=0.9, edgecolor='darkred', linewidth=0.3,
           label=f'Cases (n={len(cases)})')

ax.set_title(f"Case-control sampling (1:{CASE_CONTROL_RATIO}, seed={RANDOM_SEED})")
ax.set_xlabel("Longitude (WGS-84)")
ax.set_ylabel("Latitude (WGS-84)")
ax.legend(loc='upper left')
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig("png/controls_check.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Check image saved to png/controls_check.png")


# 7. Numerical inspection
# Verify three things:
#   1) All controls are in the sampling area
#   2) Each control is >= 200m from the nearest case
#   3) Each control is >= 100m from the nearest other controls

print("\n=== Verification ===")

# 1) sampling_area contains
in_area = sum(sampling.contains(pt) for pt in controls)
print(f"Within sampling area: {in_area} / {len(controls)}")

# 2) case-control distance
ctrl_to_case_min = []
for pt in controls:
    idx = case_tree.nearest(pt)
    d_m = pt.distance(case_points[idx]) * 111000
    ctrl_to_case_min.append(d_m)
print(f"control to nearest case distance: min={min(ctrl_to_case_min):.1f}m, "
      f"median={np.median(ctrl_to_case_min):.1f}m, max={max(ctrl_to_case_min):.1f}m")
print(f"  (All requirements are >= 200m, are they met: {min(ctrl_to_case_min) >= 200})")

# 3) control-control distance
ctrl_tree = STRtree(controls)
ctrl_to_ctrl_min = []
for i, pt in enumerate(controls):
    # Find nearest neighbor (except yourself)
    candidates = ctrl_tree.query(pt.buffer(MIN_DIST_CTRL_DEG * 3))
    candidates = [j for j in candidates if j != i]
    if not candidates:
        continue
    min_d = min(pt.distance(controls[j]) for j in candidates) * 111000
    ctrl_to_ctrl_min.append(min_d)
print(f"control to nearest control distance: min={min(ctrl_to_ctrl_min):.1f}m, "
      f"median={np.median(ctrl_to_ctrl_min):.1f}m")
print(f"  (All requirements are >= 100m, are they met: {min(ctrl_to_ctrl_min) >= 100})")