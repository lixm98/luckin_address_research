"""
Task 1: Control point sampling sensitivity experiment
Exclusion distance {0,50,100,200} m × seed {42,123,2026} = 12 runs
Each time: redraw 880 control points → recalculate 7 features → global logistic
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
from shapely.strtree import STRtree
from scipy.spatial import cKDTree
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score

# ============ CONFIG ============
SAMPLING_AREA_FILE = "geojson/sampling_area_final.geojson"
LUCKIN_FILE        = "csv/xiamen_luckin_poi_wgs84.csv"
STARBUCKS_FILE     = "csv/xiamen_starbucks_poi_wgs84.csv"
OFFICE_FILE        = "csv/xiamen_office_wgs84.csv"
RESIDENTIAL_FILE   = "csv/xiamen_residential_wgs84.csv"
MALL_FILE          = "csv/xiamen_mall_wgs84.csv"
METRO_STATION_FILE = "csv/xiamen_metro_stations_wgs84.csv"
METRO_EXIT_FILE    = "csv/xiamen_metro_exits_wgs84.csv"

EXCL_GRID = [0, 50, 100, 200]
SEED_GRID = [42, 123, 2026]
N_CONTROLS      = 880
MIN_DIST_CTRL_M = 100
MAX_ATTEMPTS    = 200_000
BUFFER_M        = 500
METRIC_CRS      = "EPSG:32650"   # UTM 50N, recorded in Appendix A

COUNT_FEATS = ["office_count", "residential_count", "mall_count", "metro_station_count"]
DIST_FEATS  = ["dist_to_nearest_luckin", "dist_to_nearest_starbucks", "dist_to_nearest_metro_exit"]
FEATURES    = COUNT_FEATS + DIST_FEATS

# ============ Loading data ============
def load_xy(path):
    """CSV → (df, projected coordinate array [n,2], unit meter)"""
    df = pd.read_csv(path)
    gs = gpd.GeoSeries(gpd.points_from_xy(df["lon_wgs84"], df["lat_wgs84"]),
                       crs="EPSG:4326").to_crs(METRIC_CRS)
    return df, np.column_stack([gs.x.values, gs.y.values])

sampling = gpd.read_file(SAMPLING_AREA_FILE).iloc[0].geometry
minx, miny, maxx, maxy = sampling.bounds

luckin_df, luckin_xy = load_xy(LUCKIN_FILE)          # 311, the distance is calculated using the whole city
luckin_names = luckin_df["name"].tolist()   # Same order as luckin_xy
cases = luckin_df[luckin_df["adname"].isin(["思明区", "湖里区"])].copy().reset_index(drop=True)
case_points = [Point(r["lon_wgs84"], r["lat_wgs84"]) for _, r in cases.iterrows()]
case_tree = STRtree(case_points)                      # Degree coordinates, consistent with the original script
assert len(cases) == 176, f"case 数 = {len(cases)}，应为176"

_, sbux_xy   = load_xy(STARBUCKS_FILE)
_, office_xy = load_xy(OFFICE_FILE)
_, resid_xy  = load_xy(RESIDENTIAL_FILE)
_, mall_xy   = load_xy(MALL_FILE)
_, mstat_xy  = load_xy(METRO_STATION_FILE)
_, mexit_xy  = load_xy(METRO_EXIT_FILE)

count_trees = {"office_count": cKDTree(office_xy), "residential_count": cKDTree(resid_xy),
               "mall_count": cKDTree(mall_xy), "metro_station_count": cKDTree(mstat_xy)}
luckin_kd = cKDTree(luckin_xy)
sbux_kd   = cKDTree(sbux_xy)
mexit_kd  = cKDTree(mexit_xy)

# ============ Sampling (the algorithm is word-for-word consistent with the original script, only parameterized) ============
def sample_controls(excl_m, seed):
    np.random.seed(seed)
    min_case_deg = excl_m / 111000
    min_ctrl_deg = MIN_DIST_CTRL_M / 111000
    controls, records, attempts = [], [], 0
    while len(controls) < N_CONTROLS and attempts < MAX_ATTEMPTS:
        attempts += 1
        lon = np.random.uniform(minx, maxx)
        lat = np.random.uniform(miny, maxy)
        pt = Point(lon, lat)
        if not sampling.contains(pt):
            continue
        i = case_tree.nearest(pt)
        if pt.distance(case_points[i]) < min_case_deg:   # Never trigger when excl_m=0
            continue
        if controls:
            if pt.distance(controls[STRtree(controls).nearest(pt)]) < min_ctrl_deg:
                continue
        controls.append(pt)
        records.append({"lon_wgs84": lon, "lat_wgs84": lat})
    if len(controls) < N_CONTROLS:
        raise RuntimeError(f"excl={excl_m} seed={seed}: 仅生成 {len(controls)} 个")
    return pd.DataFrame(records), attempts

# ============ Feature recalculation ============
def compute_features(samples):
    gs = gpd.GeoSeries(gpd.points_from_xy(samples["lon_wgs84"], samples["lat_wgs84"]),
                       crs="EPSG:4326").to_crs(METRIC_CRS)
    xy = np.column_stack([gs.x.values, gs.y.values])
    out = samples.copy()
    for col, tree in count_trees.items():
        out[col] = [len(v) for v in tree.query_ball_point(xy, r=BUFFER_M)]
    # Dist to nearest luckin: Replicate the original rules -all the same names are excluded, and the nearest store with the same name is taken.
    K = 8                                    # Maximum of 3 duplicate names, 8 is enough
    dists, idxs = luckin_kd.query(xy, k=K)
    names = samples["name"].values
    d_luckin = np.full(len(out), np.nan)
    for i in range(len(out)):
        for k in range(K):
            if luckin_names[idxs[i, k]] != names[i]:
                d_luckin[i] = dists[i, k]
                break
    out["dist_to_nearest_luckin"]     = d_luckin
    out["dist_to_nearest_starbucks"]  = sbux_kd.query(xy, k=1)[0]
    out["dist_to_nearest_metro_exit"] = mexit_kd.query(xy, k=1)[0]
    return out

# ============ Global logistic ============
def run_logit(feat):
    df = feat.copy()
    zcols = []
    for c in DIST_FEATS:                       # The distance variable is rez-normalized according to this sample (ddof=1)
        df[c + "_z"] = (df[c] - df[c].mean()) / df[c].std(ddof=0)
        zcols.append(c + "_z")
    cols = COUNT_FEATS + zcols
    X = sm.add_constant(df[cols].astype(float))
    y = df["case"].astype(int)
    m = sm.Logit(y, X).fit(disp=0, maxiter=100)
    auc = roc_auc_score(y, m.predict(X))
    mcf = 1 - m.llf / m.llnull
    return m, auc, mcf, cols

# ============ Main loop: 12 runs ============
rows = []
for excl in EXCL_GRID:
    for seed in SEED_GRID:
        ctrl, attempts = sample_controls(excl, seed)
        ctrl["case"] = 0
        samp = pd.concat([cases[["lon_wgs84", "lat_wgs84", "name"]].assign(case=1),
                  ctrl.assign(name="(control)")], ignore_index=True)
        feat = compute_features(samp)
        feat.to_csv(f"csv/task1_features_excl{excl}_seed{seed}.csv",
                    index=False, encoding="utf-8-sig")
        m, auc, mcf, cols = run_logit(feat)
        row = {"excl_m": excl, "seed": seed, "n_controls": len(ctrl),
               "attempts": attempts, "AUC": round(auc, 4), "McFadden_R2": round(mcf, 4)}
        for base, col in zip(FEATURES, cols):
            row[f"{base}_coef"] = round(m.params[col], 4)
            row[f"{base}_p"]    = round(m.pvalues[col], 4)
            row[f"{base}_OR"]   = round(np.exp(m.params[col]), 4)
        rows.append(row)
        print(f"excl={excl:>3} seed={seed:>4} | AUC={auc:.3f} | "
              f"office β={row['office_count_coef']:+.3f} (p={row['office_count_p']:.3f}) | "
              f"dist_luckin β={row['dist_to_nearest_luckin_coef']:+.3f}")

summary = pd.DataFrame(rows)
summary.to_csv("csv/task1_sensitivity_summary.csv", index=False, encoding="utf-8-sig")
print("\n✓ saved csv/task1_sensitivity_summary.csv")

# ============ Decision Rule Readout ============
print("\n=== Key variable stability (12 runs)===")
for f in ["office_count", "dist_to_nearest_starbucks", "dist_to_nearest_luckin"]:
    c, p = summary[f + "_coef"], summary[f + "_p"]
    print(f"{f}: coef ∈ [{c.min():.3f}, {c.max():.3f}], Symbol consistent={np.sign(c).nunique()==1}, "
          f"p<0.05 frequency={(p < 0.05).sum()}/12")
print("\ndist_to_nearest_luckin The coefficient drifts with the excluded gear:")
print(summary.groupby("excl_m")["dist_to_nearest_luckin_coef"]
      .agg(["mean", "min", "max"]).round(3))
print(f"\nAUC scope:{summary['AUC'].min():.3f} – {summary['AUC'].max():.3f}")