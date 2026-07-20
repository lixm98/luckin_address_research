"""
Starbucks Logistic Regression (Control Model)
=========================================
Exactly the same settings as Luckin:
  -7 features
  -case-control 1:5
  -200m exclusion radius /100m control spacing
  -Distance feature normalization
  -Random seed 42
The only difference: case = Starbucks store, not Luckin store
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import statsmodels.api as sm
from shapely.geometry import Point
from shapely.strtree import STRtree
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


# 0. Parameters (same as Luckin)
RANDOM_SEED = 42
CASE_CONTROL_RATIO = 5
MIN_DIST_CASE_M = 200
MIN_DIST_CTRL_M = 100
BUFFER_M = 500

MIN_DIST_CASE_DEG = MIN_DIST_CASE_M / 111000
MIN_DIST_CTRL_DEG = MIN_DIST_CTRL_M / 111000

WGS84 = "EPSG:4326"
UTM = "EPSG:32650"

np.random.seed(RANDOM_SEED)


# 1. Load cases + sampling area in Starbucks island
sb_all = pd.read_csv("xiamen_starbucks_poi_wgs84.csv")
sb_cases = sb_all[sb_all['adname'].isin(['思明区', '湖里区'])].copy().reset_index(drop=True)
N_CASES = len(sb_cases)
N_CONTROLS = N_CASES * CASE_CONTROL_RATIO

print(f"✓ Starbucks on the island cases: {N_CASES}")
print(f"✓ Target controls:    {N_CONTROLS}")

sampling = gpd.read_file("sampling_area_final.geojson").iloc[0].geometry


# 2. Spread Starbucks controls (independent sampling, no reuse of Luckin)
case_points = [Point(r['lon_wgs84'], r['lat_wgs84']) for _, r in sb_cases.iterrows()]
case_tree = STRtree(case_points)

controls = []
control_records = []
attempts = 0
minx, miny, maxx, maxy = sampling.bounds

while len(controls) < N_CONTROLS and attempts < 100_000:
    attempts += 1
    lon = np.random.uniform(minx, maxx)
    lat = np.random.uniform(miny, maxy)
    pt = Point(lon, lat)

    if not sampling.contains(pt):
        continue

    nearest_case_idx = case_tree.nearest(pt)
    if pt.distance(case_points[nearest_case_idx]) < MIN_DIST_CASE_DEG:
        continue

    if controls:
        ctrl_tree = STRtree(controls)
        nearest_ctrl_idx = ctrl_tree.nearest(pt)
        if pt.distance(controls[nearest_ctrl_idx]) < MIN_DIST_CTRL_DEG:
            continue

    controls.append(pt)
    control_records.append({"lon_wgs84": lon, "lat_wgs84": lat, "case": 0, "name": "(control)"})

    if len(controls) % 50 == 0:
        print(f"  Generated {len(controls)}/{N_CONTROLS}")

print(f"✓ Total attempts: {attempts}, Acceptance rate: {len(controls)/attempts*100:.1f}%")


# 3. Merge case + control
cases_out = sb_cases[['lon_wgs84', 'lat_wgs84', 'name']].copy()
cases_out['case'] = 1
controls_out = pd.DataFrame(control_records)
samples_sb = pd.concat([cases_out, controls_out], ignore_index=True)[
    ['lon_wgs84', 'lat_wgs84', 'case', 'name']
]
print(f"✓ samples merged: {len(samples_sb)} rows "
      f"({samples_sb['case'].sum()} cases + {(samples_sb['case']==0).sum()} controls)")


# 4. Feature engineering (function consistent with Luckin)
samples_gdf = gpd.GeoDataFrame(
    samples_sb,
    geometry=[Point(r['lon_wgs84'], r['lat_wgs84']) for _, r in samples_sb.iterrows()],
    crs=WGS84
).to_crs(UTM)


def load_pois(csv_path):
    df = pd.read_csv(csv_path)
    return gpd.GeoDataFrame(
        df,
        geometry=[Point(r['lon_wgs84'], r['lat_wgs84']) for _, r in df.iterrows()],
        crs=WGS84
    ).to_crs(UTM)


print("Loading POI data...")
office = load_pois("xiamen_office_wgs84.csv")
residential = load_pois("xiamen_residential_wgs84.csv")
mall = load_pois("xiamen_mall_wgs84.csv")
metro_st = load_pois("xiamen_metro_stations_wgs84.csv")
metro_exits = load_pois("xiamen_metro_exits_wgs84.csv")
luckin_all = load_pois("xiamen_luckin_poi_wgs84.csv")
starbucks_all = load_pois("xiamen_starbucks_poi_wgs84.csv")


def count_pois_in_buffer(sample_points, poi_gdf, buffer_m):
    poi_points = list(poi_gdf.geometry)
    tree = STRtree(poi_points)
    counts = []
    for pt in sample_points:
        candidates = tree.query(pt.buffer(buffer_m))
        n = sum(1 for i in candidates if pt.distance(poi_points[i]) <= buffer_m)
        counts.append(n)
    return counts


def dist_to_nearest(sample_points, poi_gdf, exclude_self_name=None, sample_names=None):
    poi_points = list(poi_gdf.geometry)
    poi_names = poi_gdf['name'].tolist() if 'name' in poi_gdf.columns else [None] * len(poi_gdf)
    tree = STRtree(poi_points)
    distances = []
    for i, pt in enumerate(sample_points):
        nearest_idx = tree.nearest(pt)
        if exclude_self_name and sample_names is not None:
            sname = sample_names.iloc[i]
            if poi_names[nearest_idx] == sname:
                candidates = tree.query(pt.buffer(5000))
                candidates = [j for j in candidates if poi_names[j] != sname]
                if candidates:
                    nearest_idx = min(candidates, key=lambda j: pt.distance(poi_points[j]))
                else:
                    distances.append(np.nan)
                    continue
        distances.append(pt.distance(poi_points[nearest_idx]))
    return distances


print("Calculating features...")
samples_gdf['office_count']        = count_pois_in_buffer(samples_gdf.geometry, office, BUFFER_M)
samples_gdf['residential_count']   = count_pois_in_buffer(samples_gdf.geometry, residential, BUFFER_M)
samples_gdf['mall_count']          = count_pois_in_buffer(samples_gdf.geometry, mall, BUFFER_M)
samples_gdf['metro_station_count'] = count_pois_in_buffer(samples_gdf.geometry, metro_st, BUFFER_M)

# Note: Starbucks case excludes itself when counting "the nearest Starbucks"; there is no need to exclude itself when counting the "nearest Luckin"
samples_gdf['dist_to_nearest_luckin']      = dist_to_nearest(samples_gdf.geometry, luckin_all)
samples_gdf['dist_to_nearest_starbucks']   = dist_to_nearest(
    samples_gdf.geometry, starbucks_all,
    exclude_self_name=True, sample_names=samples_gdf['name']
)
samples_gdf['dist_to_nearest_metro_exit']  = dist_to_nearest(samples_gdf.geometry, metro_exits)

samples_gdf_wgs = samples_gdf.to_crs(WGS84)
features_sb = pd.DataFrame({
    "lon_wgs84":                    samples_gdf_wgs.geometry.x,
    "lat_wgs84":                    samples_gdf_wgs.geometry.y,
    "case":                         samples_gdf['case'].values,
    "name":                         samples_gdf['name'].values,
    "office_count":                 samples_gdf['office_count'].values,
    "residential_count":            samples_gdf['residential_count'].values,
    "mall_count":                   samples_gdf['mall_count'].values,
    "metro_station_count":          samples_gdf['metro_station_count'].values,
    "dist_to_nearest_luckin":       samples_gdf['dist_to_nearest_luckin'].values,
    "dist_to_nearest_starbucks":    samples_gdf['dist_to_nearest_starbucks'].values,
    "dist_to_nearest_metro_exit":   samples_gdf['dist_to_nearest_metro_exit'].values,
})
features_sb.to_csv("starbucks_features.csv", index=False, encoding="utf-8-sig")
print(f"✓ Features saved to starbucks_features.csv ({len(features_sb)} rows)")


# 5. Run Logistic
FEATURES = [
    'office_count', 'residential_count', 'mall_count', 'metro_station_count',
    'dist_to_nearest_luckin', 'dist_to_nearest_starbucks', 'dist_to_nearest_metro_exit',
]
X = features_sb[FEATURES].copy()
y = features_sb['case'].astype(int)

dist_cols = [c for c in FEATURES if c.startswith('dist_')]
X[dist_cols] = StandardScaler().fit_transform(X[dist_cols])

X_const = sm.add_constant(X)
sb_logit = sm.Logit(y, X_const).fit(disp=False)

print("\n" + "=" * 80)
print("Starbucks Logistic Regression Results")
print("=" * 80)
print(sb_logit.summary())


# 6. Save the coefficient table -containing all fields required for comparison
# Required fields: coef, std_err, p_value, OR, OR_CI_low, OR_CI_high
# The KeyError in the previous code comparison was due to the lack of std_err

sb_conf = sb_logit.conf_int()
sb_conf.columns = ['CI_lower', 'CI_upper']

sb_table = pd.DataFrame({
    'coef':       sb_logit.params,
    'std_err':    sb_logit.bse,        # ★ Key fields
    'z_value':    sb_logit.tvalues,
    'p_value':    sb_logit.pvalues,
    'CI_lower':   sb_conf['CI_lower'],   # log-odds scale
    'CI_upper':   sb_conf['CI_upper'],
    'OR':         np.exp(sb_logit.params),
    'OR_CI_low':  np.exp(sb_conf['CI_lower']),
    'OR_CI_high': np.exp(sb_conf['CI_upper']),
})
sb_table.to_csv("starbucks_logit_coefficients.csv", encoding='utf-8-sig')
print(f"\n✓ Features saved to starbucks_logit_coefficients.csv")
print(f"  Fields: {list(sb_table.columns)}")


# 7. Overall performance of the model
sb_auc = roc_auc_score(y, sb_logit.predict(X_const))
n_obs = len(y)
n_cases = int(y.sum())
n_controls = n_obs - n_cases

# Ai cc
k = sb_logit.df_model + 1
sb_aicc = sb_logit.aic + (2 * k * (k + 1)) / (n_obs - k - 1)

summary = pd.DataFrame({
    'metric': ['AUC', 'Pseudo R² (McFadden)', 'AIC', 'AICc', 'BIC', 'Log-likelihood',
               'N obs', 'N cases', 'N controls', 'LLR p-value'],
    'value':  [sb_auc, sb_logit.prsquared, sb_logit.aic, sb_aicc, sb_logit.bic,
               sb_logit.llf, n_obs, n_cases, n_controls, sb_logit.llr_pvalue],
})
summary.to_csv("starbucks_model_summary.csv", index=False, encoding='utf-8-sig')

print(f"\n=== Starbucks Model Performance ===")
print(f"  AUC               = {sb_auc:.4f}")
print(f"  Pseudo R²         = {sb_logit.prsquared:.4f}")
print(f"  AICc              = {sb_aicc:.2f}")
print(f"  N obs / cases     = {n_obs} / {n_cases}")
print(f"✓ Features saved to starbucks_model_summary.csv")