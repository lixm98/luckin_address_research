"""GWLR — 50m final data version"""
import pandas as pd, numpy as np, warnings
from mgwr.gwr import GWR
from spglm.family import Binomial
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import geopandas as gpd
import statsmodels.api as sm
warnings.filterwarnings('ignore')

df = pd.read_csv("task1_features_excl50_seed42.csv")
print(f"✓ Loaded: {len(df)} rows")

gs = gpd.GeoSeries(gpd.points_from_xy(df['lon_wgs84'], df['lat_wgs84']),
                   crs="EPSG:4326").to_crs("EPSG:32650")
coords = np.column_stack([gs.x.values, gs.y.values])

# ---metro_station_count local zero variance check (within the window of minimum candidate bandwidth K=400) ---
_, nn = cKDTree(coords).query(coords, k=400)
metro = df['metro_station_count'].values
n_zero = int(sum(metro[nn[i]].std() == 0 for i in range(len(df))))
print(f"metro_station_count zero variance windows: {n_zero}/{len(df)}")

FEATURES = ['office_count', 'residential_count', 'mall_count',
            'dist_to_nearest_luckin', 'dist_to_nearest_starbucks',
            'dist_to_nearest_metro_exit']
if n_zero == 0:
    FEATURES.insert(3, 'metro_station_count')
    print("→ No zero variance window, included this time metro_station_count")
else:
    print("→ There are zero variance windows, excluding metro_station_count (consistent with original setting)")

X = df[FEATURES].copy()
dist_cols = [c for c in FEATURES if c.startswith('dist_')]
X[dist_cols] = StandardScaler().fit_transform(X[dist_cols])
X_arr = X.values
y = df['case'].astype(int).values.reshape(-1, 1)

# ---Bandwidth search (grid is consistent with original settings) ---
print("\n=== Manual testing different bandwidth ===")
results = {}
for bw in [400, 500, 600, 700, 800]:
    res = GWR(coords, y, X_arr, bw=bw, kernel='bisquare',
              fixed=False, family=Binomial(), n_jobs=1).fit()
    auc = roc_auc_score(y.flatten(), res.predy.flatten())
    results[bw] = {'aicc': res.aicc, 'auc': auc, 'res': res}
    print(f"  K={bw}: AICc={res.aicc:.2f}, AUC={auc:.4f}")

best_bw = min(results, key=lambda k: results[k]['aicc'])
gwlr_results, gwlr_aicc, gwlr_auc = (results[best_bw]['res'],
                                     results[best_bw]['aicc'], results[best_bw]['auc'])
print(f"\n✓ Optimal bandwidth: K = {best_bw}")

# ---Compare logistic (same feature set, ensuring AICc is comparable) ---
X_const = sm.add_constant(X)
logit = sm.Logit(y, X_const).fit(disp=False)
logit_aicc = logit.aic + (2*(logit.df_model+1)*(logit.df_model+2)) / (len(y)-logit.df_model-2)
logit_auc = roc_auc_score(y.flatten(), logit.predict(X_const))
print("\n=== Model Comparison (in-sample, 50m data) ===")
print(f"AICc:  logistic {logit_aicc:.2f} | GWLR {gwlr_aicc:.2f} | Improvement {logit_aicc-gwlr_aicc:+.2f}")
print(f"AUC:   logistic {logit_auc:.4f} | GWLR {gwlr_auc:.4f} | Improvement {gwlr_auc-logit_auc:+.4f}")

# # ---Coefficient space statistics ---
# lp, lt = gwlr_results.params, gwlr_results.tvalues
# print("\n=== Coefficient space statistics ===")
# print(f"{'feature':<30}{'min':>8}{'max':>8}{'median':>9}{'sig %':>8}")
# for i, f in enumerate(FEATURES):
#     c, t = lp[:, i+1], lt[:, i+1]
#     print(f"{f:<30}{c.min():>8.3f}{c.max():>8.3f}{np.median(c):>9.3f}"
#           f"{(np.abs(t)>1.96).mean()*100:>7.1f}%")

# ---Coefficient space statistics ---
lp, lt = gwlr_results.params, gwlr_results.tvalues

# ---metro_station_count diagnostic output ---
metro_idx = FEATURES.index('metro_station_count') + 1
metro_t = lt[:, metro_idx]
metro_coef = lp[:, metro_idx]

print("\n=== metro_station_count diagnostic ===")
print("Value distribution:")
print(df['metro_station_count'].value_counts().sort_index())
print(f"Local coefficient range: {metro_coef.min():.6f} to {metro_coef.max():.6f}")
print(f"Local t-value range: {metro_t.min():.6f} to {metro_t.max():.6f}")
print(f"Max |t|: {np.abs(metro_t).max():.6f}")
print(f"n(|t| > 1.96): {(np.abs(metro_t) > 1.96).sum()}/{len(df)}")

print("\n=== Coefficient space statistics ===")
print(f"{'feature':<30}{'min':>8}{'max':>8}{'median':>9}{'sig %':>8}")
for i, f in enumerate(FEATURES):
    c, t = lp[:, i+1], lt[:, i+1]
    print(f"{f:<30}{c.min():>8.3f}{c.max():>8.3f}{np.median(c):>9.3f}"
          f"{(np.abs(t)>1.96).mean()*100:>7.1f}%")

# ---Local collinearity diagnosis: weighted VIF + condition number per point (core/bandwidth consistent with GWLR) ---
print("\n=== Local collinearity diagnosis (weighted VIF / condition number, K=%d) ===" % best_bw)
p_ = len(FEATURES)
vifs = np.full((len(df), p_), np.nan)
cns  = np.full(len(df), np.nan)
Xv = X.values.astype(float)
for i in range(len(df)):
    d = np.hypot(coords[:, 0] - coords[i, 0], coords[:, 1] - coords[i, 1])
    h = np.sort(d)[best_bw - 1]                      # Adaptive bandwidth = kth closest point distance
    w = np.where(d < h, (1 - (d / h) ** 2) ** 2, 0.0)  # Bisquare
    m = w > 0
    Xw, ww = Xv[m], w[m]
    mu = np.average(Xw, axis=0, weights=ww)
    Xc = Xw - mu
    cov = (Xc * ww[:, None]).T @ Xc / ww.sum()       # weighted covariance
    sd = np.sqrt(np.diag(cov))
    if (sd == 0).any():                              # Local zero variance point: diagnostic record na n
        continue
    R = cov / np.outer(sd, sd)                       # weighted correlation matrix
    eig = np.linalg.eigvalsh(R)
    cns[i] = np.sqrt(eig.max() / max(eig.min(), 1e-12))
    vifs[i] = np.diag(np.linalg.inv(R))

print(f"{'feature':<30}{'VIF median':>9}{'VIF max':>9}{'>5 ratio':>8}{'>10 ratio':>9}")
for j, f in enumerate(FEATURES):
    v = vifs[:, j][~np.isnan(vifs[:, j])]
    print(f"{f:<30}{np.median(v):>9.2f}{v.max():>9.2f}"
          f"{(v > 5).mean()*100:>7.1f}%{(v > 10).mean()*100:>8.1f}%")
cn_v = cns[~np.isnan(cns)]
print(f"Condition number: median={np.median(cn_v):.1f}, max={cn_v.max():.1f}, "
      f">30 ratio={(cn_v > 30).mean()*100:.1f}%")
n_nan = int(np.isnan(cns).sum())
if n_nan: print(f"⚠️ {n_nan} points excluded due to local zero variance")

# ---Save local coefficients ---
out = df[['lon_wgs84', 'lat_wgs84', 'case']].copy()
out['intercept'] = lp[:, 0]
out['local_condition_number'] = cns
for i, f in enumerate(FEATURES):
    out[f'coef_{f}'] = lp[:, i+1]
    out[f'tval_{f}'] = lt[:, i+1]
    out[f'vif_{f}']  = vifs[:, i]
out.to_csv("gwlr_local_coefficients_excl50.csv", index=False, encoding='utf-8-sig')
print("\n✓ gwlr_local_coefficients_excl50.csv has been saved (with local VIF and condition number)")

# ---Coefficient map ---
n = len(FEATURES)
fig, axes = plt.subplots(2, 4, figsize=(20, 10)); axes = axes.flatten()
for i, f in enumerate(FEATURES):
    ax, c, t = axes[i], lp[:, i+1], lt[:, i+1]
    vmax = max(abs(c.min()), abs(c.max())); sig = np.abs(t) > 1.96
    ax.scatter(df['lon_wgs84'][~sig], df['lat_wgs84'][~sig], c='lightgray', s=10, alpha=0.4)
    sc = ax.scatter(df['lon_wgs84'][sig], df['lat_wgs84'][sig], c=c[sig],
                    cmap='RdBu_r', s=25, vmin=-vmax, vmax=vmax)
    ax.set_title(f"{f}\n(sig: {sig.sum()}/{len(sig)})", fontsize=10)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(sc, ax=ax, shrink=0.7)
for j in range(n, 8): axes[j].axis('off')
# plt.suptitle(f"GWLR Local Coefficients (K={best_bw}, 50m dataset)", y=1.00)
plt.tight_layout()
plt.savefig("gwlr_coefficient_maps_excl50.png", dpi=150, bbox_inches='tight')
plt.close()

# ---ROC comparison ---
fpr_l, tpr_l, _ = roc_curve(y.flatten(), logit.predict(X_const))
fpr_g, tpr_g, _ = roc_curve(y.flatten(), gwlr_results.predy.flatten())
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr_l, tpr_l, lw=2, label=f'Logistic (AUC={logit_auc:.3f})', color='steelblue')
ax.plot(fpr_g, tpr_g, lw=2, label=f'GWLR K={best_bw} (AUC={gwlr_auc:.3f})', color='crimson')
ax.plot([0,1],[0,1],'k--',lw=1,alpha=0.5)
ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend(loc='lower right'); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("roc_logistic_vs_gwlr_excl50.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ Picture saved")