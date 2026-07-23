"""
Starbucks control point redraw (50m exclusion, seed 42) + feature recalculation
"""
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
from shapely.strtree import STRtree
from scipy.spatial import cKDTree

SEED, EXCL_M, MIN_CTRL_M = 42, 50, 100
N_CONTROLS, MAX_ATTEMPTS = 300, 200_000
BUFFER_M, UTM = 500, "EPSG:32650"

def load_xy(path):
    df = pd.read_csv(path)
    gs = gpd.GeoSeries(gpd.points_from_xy(df["lon_wgs84"], df["lat_wgs84"]),
                       crs="EPSG:4326").to_crs(UTM)
    return df, np.column_stack([gs.x.values, gs.y.values])

sampling = gpd.read_file("geojson/sampling_area_final.geojson").iloc[0].geometry
minx, miny, maxx, maxy = sampling.bounds

sbux_df, sbux_xy = load_xy("csv/xiamen_starbucks_poi_wgs84.csv")     # 94 homes, distance from pool
sbux_names = sbux_df["name"].tolist()
cases = sbux_df[sbux_df["adname"].isin(["思明区", "湖里区"])].copy().reset_index(drop=True)
assert len(cases) == 60, f"case数={len(cases)}"
case_points = [Point(r["lon_wgs84"], r["lat_wgs84"]) for _, r in cases.iterrows()]
case_tree = STRtree(case_points)

luckin_df, luckin_xy = load_xy("csv/xiamen_luckin_poi_wgs84.csv")
_, office_xy = load_xy("csv/xiamen_office_wgs84.csv")
_, resid_xy  = load_xy("csv/xiamen_residential_wgs84.csv")
_, mall_xy   = load_xy("csv/xiamen_mall_wgs84.csv")
_, mstat_xy  = load_xy("csv/xiamen_metro_stations_wgs84.csv")
_, mexit_xy  = load_xy("csv/xiamen_metro_exits_wgs84.csv")
count_trees = {"office_count": cKDTree(office_xy), "residential_count": cKDTree(resid_xy),
               "mall_count": cKDTree(mall_xy), "metro_station_count": cKDTree(mstat_xy)}
luckin_kd, sbux_kd, mexit_kd = cKDTree(luckin_xy), cKDTree(sbux_xy), cKDTree(mexit_xy)

# ----Sampling (same algorithm as Luckin task 1, 50m) ----
np.random.seed(SEED)
min_case_deg, min_ctrl_deg = EXCL_M/111000, MIN_CTRL_M/111000
controls, records, attempts = [], [], 0
while len(controls) < N_CONTROLS and attempts < MAX_ATTEMPTS:
    attempts += 1
    lon, lat = np.random.uniform(minx, maxx), np.random.uniform(miny, maxy)
    pt = Point(lon, lat)
    if not sampling.contains(pt): continue
    if pt.distance(case_points[case_tree.nearest(pt)]) < min_case_deg: continue
    if controls and pt.distance(controls[STRtree(controls).nearest(pt)]) < min_ctrl_deg: continue
    controls.append(pt); records.append({"lon_wgs84": lon, "lat_wgs84": lat})
assert len(controls) == N_CONTROLS, f"仅生成{len(controls)}个"
print(f"✓ 300 controls, 尝试{attempts}次")

samp = pd.concat([cases[["lon_wgs84","lat_wgs84","name"]].assign(case=1),
                  pd.DataFrame(records).assign(case=0, name="(control)")],
                 ignore_index=True)

# ----Features (the caliber is the same as the Luckin side; self-exclusion and switch to the Starbucks side) ----
gs = gpd.GeoSeries(gpd.points_from_xy(samp["lon_wgs84"], samp["lat_wgs84"]),
                   crs="EPSG:4326").to_crs(UTM)
xy = np.column_stack([gs.x.values, gs.y.values])
for col, tree in count_trees.items():
    samp[col] = [len(v) for v in tree.query_ball_point(xy, r=BUFFER_M)]

K = 8
dists, idxs = sbux_kd.query(xy, k=K)
d_sb = np.full(len(samp), np.nan)
for i in range(len(samp)):
    for k in range(K):
        if sbux_names[idxs[i, k]] != samp["name"].iloc[i]:
            d_sb[i] = dists[i, k]; break
samp["dist_to_nearest_starbucks"] = d_sb
samp["dist_to_nearest_luckin"]     = luckin_kd.query(xy, k=1)[0]
samp["dist_to_nearest_metro_exit"] = mexit_kd.query(xy, k=1)[0]

assert not samp.isna().any().any()
samp.to_csv("csv/task3_starbucks_features_excl50_seed42.csv", index=False, encoding="utf-8-sig")
print(f"✓ saved csv/task3_starbucks_features_excl50_seed42.csv ({len(samp)}rows = "
      f"{samp['case'].sum()} cases + {(samp['case']==0).sum()} controls)")
print("\ncase vs control Mean (original units):")
print(samp.groupby("case")[list(count_trees)+['dist_to_nearest_luckin',
      'dist_to_nearest_starbucks','dist_to_nearest_metro_exit']].mean().round(1))