import pandas as pd, numpy as np, warnings
import geopandas as gpd
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy.spatial import cKDTree
import xgboost as xgb
from mgwr.gwr import GWR
from spglm.family import Binomial
warnings.filterwarnings('ignore')

SEED, K_BLOCKS = 42, 5
BW_CANDIDATES = [400, 500, 600, 700, 800]   # Consistent with original GWLR grid
UTM = "EPSG:32650"
FEATURES = ['office_count','residential_count','mall_count','metro_station_count',
            'dist_to_nearest_luckin','dist_to_nearest_starbucks','dist_to_nearest_metro_exit']
GWLR_FEATS = FEATURES.copy()
DIST_COLS  = [c for c in FEATURES if c.startswith('dist_')]

# ----------data ----------
df = pd.read_csv("csv/task1_features_excl50_seed42.csv")
def to_utm(lon, lat):
    gs = gpd.GeoSeries(gpd.points_from_xy(lon, lat), crs="EPSG:4326").to_crs(UTM)
    return np.column_stack([gs.x.values, gs.y.values])
xy = to_utm(df['lon_wgs84'], df['lat_wgs84'])
y_all = df['case'].astype(int).values
pool_xy = to_utm(*pd.read_csv("csv/xiamen_luckin_poi_wgs84.csv")[['lon_wgs84','lat_wgs84']].values.T)  # 311 homes

# ----------Chunking ----------
blocks = KMeans(n_clusters=K_BLOCKS, random_state=SEED, n_init=10).fit_predict(xy)
print("block composition:")
for b in range(K_BLOCKS):
    m = blocks == b
    print(f"  block {b}: n={m.sum()}, cases={y_all[m].sum()}")

# ----------GWLR fold-out prediction: locally weighted logistic centered on the test point ----------
def gwlr_predict_point(pt, tr_xy, X_tr_const, y_tr, K, x_rows):
    d = np.hypot(tr_xy[:,0]-pt[0], tr_xy[:,1]-pt[1])
    h = np.sort(d)[K-1]                              # Adaptive bandwidth = kth nearest training point
    w = np.where(d < h, (1-(d/h)**2)**2, 0.0)        # Bisquare
    m = w > 0
    fit = sm.GLM(y_tr[m], X_tr_const[m], family=sm.families.Binomial(),
                 var_weights=w[m]).fit()
    return [float(fit.predict(x.reshape(1,-1))[0]) for x in x_rows]

# ----------Main loop ----------
rows = []
for b in range(K_BLOCKS):
    te = blocks == b; tr = ~te
    print(f"\n=== fold {b} (test n={te.sum()}) ===")
    scaler = StandardScaler().fit(df.loc[tr, DIST_COLS])      # Normalization only fits within the training fold
    def make_X(mask, luckin_override=None):
        X = df.loc[mask, FEATURES].copy()
        if luckin_override is not None:
            X['dist_to_nearest_luckin'] = luckin_override
        X[DIST_COLS] = scaler.transform(X[DIST_COLS])
        return X.reset_index(drop=True)
    X_tr, X_te_dec = make_X(tr), make_X(te)
    # Strict: Eliminate case stores in the test block (match by coordinates, tolerance 1m), and recalculate the nearest Luckin distance to the test point
    case_te_xy = xy[te & (y_all == 1)]
    keep = cKDTree(case_te_xy).query(pool_xy, k=1)[0] > 1.0 if len(case_te_xy) \
           else np.ones(len(pool_xy), bool)
    X_te_str = make_X(te, luckin_override=cKDTree(pool_xy[keep]).query(xy[te], k=1)[0])
    y_tr_v, y_te_v = y_all[tr], y_all[te]
    versions = [('declared', X_te_dec), ('strict', X_te_str)]

    # 1) Logistic
    lg = sm.Logit(y_tr_v, sm.add_constant(X_tr, has_constant='add')).fit(disp=0)
    for ver, X_te in versions:
        p = lg.predict(sm.add_constant(X_te, has_constant='add'))
        rows.append(dict(fold=b, model='logistic', version=ver, auc=roc_auc_score(y_te_v, p)))

    # 2) XGBoost (the super parameters are fixed to the values ​​in Chapter 4, no parameter adjustment is required)
    xm = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                           subsample=0.8, colsample_bytree=0.8, scale_pos_weight=5,
                           eval_metric='auc', random_state=SEED)
    xm.fit(X_tr, y_tr_v)
    for ver, X_te in versions:
        rows.append(dict(fold=b, model='xgboost', version=ver,
                         auc=roc_auc_score(y_te_v, xm.predict_proba(X_te)[:,1])))

    # 3) GWLR: The bandwidth is selected according to AICc within the training fold, and local weighted logistic is used for prediction.
    Xg_tr = X_tr[GWLR_FEATS].values
    best_K, best_aicc = None, np.inf
    for K in BW_CANDIDATES:
        r = GWR(xy[tr], y_tr_v.reshape(-1,1), Xg_tr, bw=K, kernel='bisquare',
                fixed=False, family=Binomial(), n_jobs=1).fit()
        print(f"  GWLR K={K}: AICc={r.aicc:.1f}")
        if r.aicc < best_aicc: best_K, best_aicc = K, r.aicc
    print(f"  → select K={best_K}")
    Xg_tr_const = np.column_stack([np.ones(tr.sum()), Xg_tr])
    fb = sm.Logit(y_tr_v, Xg_tr_const).fit(disp=0)            # What to know when local fitting fails
    p_dec, p_str, n_fb = [], [], 0
    for j, i in enumerate(np.where(te)[0]):
        x_d = np.r_[1.0, X_te_dec.loc[j, GWLR_FEATS].values]
        x_s = np.r_[1.0, X_te_str.loc[j, GWLR_FEATS].values]
        try:
            pd_, ps_ = gwlr_predict_point(xy[i], xy[tr], Xg_tr_const, y_tr_v, best_K, [x_d, x_s])
        except Exception:
            n_fb += 1
            pd_, ps_ = float(fb.predict(x_d.reshape(1,-1))[0]), float(fb.predict(x_s.reshape(1,-1))[0])
        p_dec.append(pd_); p_str.append(ps_)
    if n_fb: print(f"  Local fitting gives the bottom line {n_fb} point")
    rows.append(dict(fold=b, model='gwlr', version='declared', auc=roc_auc_score(y_te_v, p_dec), bw=best_K))
    rows.append(dict(fold=b, model='gwlr', version='strict',   auc=roc_auc_score(y_te_v, p_str), bw=best_K))

# ----------Summary ----------
res = pd.DataFrame(rows)
res.to_csv("csv/task2_blocked_cv_results.csv", index=False, encoding="utf-8-sig")
print("\n=== New Table 4.7: blocked CV AUC (mean ± sd, 5 folds) ===")
print(res.groupby(['model','version'])['auc'].agg(['mean','std']).round(4))
print("\nPer-fold details saved to csv/task2_blocked_cv_results.csv")