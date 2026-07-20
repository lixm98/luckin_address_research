"""
GWLR -manual bandwidth version (bypassing mgwr automatic search bug)
"""

import pandas as pd
import numpy as np
from mgwr.gwr import GWR
from spglm.family import Binomial
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')


# 1. Load data (same as above)
df = pd.read_csv("features.csv")
print(f"✓ Loaded: {len(df)} rows")

FEATURES = [
    'office_count', 
    'residential_count', 
    'mall_count', 
    # 'metro_station_count',
    'dist_to_nearest_luckin', 
    'dist_to_nearest_starbucks', 
    'dist_to_nearest_metro_exit',
]
X_raw = df[FEATURES].copy()
y = df['case'].astype(int).values.reshape(-1, 1)

dist_cols = [c for c in FEATURES if c.startswith('dist_')]
X = X_raw.copy()
X[dist_cols] = StandardScaler().fit_transform(X[dist_cols])
X_arr = X.values

gdf = gpd.GeoDataFrame(
    df,
    geometry=[Point(r['lon_wgs84'], r['lat_wgs84']) for _, r in df.iterrows()],
    crs="EPSG:4326"
).to_crs("EPSG:32650")
coords = np.array([(p.x, p.y) for p in gdf.geometry])


# 2. Manually test several bandwidths and choose the one with the lowest AICc
# There is a bug in the automatic search. We manually try 5 candidate K and choose the best one.
# Candidate range: 120, 180, 240, 320, 450
#   -120 is close to the lower limit (enough samples per window)
#   -450 is close to the upper limit (to avoid degenerating into global)
#   -Several grid-like tests in the middle

print("\n=== Manually testing different bandwidths ===")
print("（each takes about 30-60 seconds）")

candidates = [400, 500, 600, 700, 800]
results = {}

for bw in candidates:
    print(f"\n  Running K = {bw} ...")
    model = GWR(coords, y, X_arr, bw=bw, kernel='bisquare',
                fixed=False, family=Binomial(), n_jobs=1)  # n_jobs=1 turns off parallelism
    res = model.fit()
    aicc = res.aicc
    auc  = roc_auc_score(y.flatten(), res.predy.flatten())
    results[bw] = {'aicc': aicc, 'auc': auc, 'res': res}
    print(f"    AICc = {aicc:.2f}, AUC = {auc:.4f}")

# Choose the one with the lowest AICc
best_bw = min(results, key=lambda k: results[k]['aicc'])
gwlr_results = results[best_bw]['res']
gwlr_aicc = results[best_bw]['aicc']
gwlr_auc = results[best_bw]['auc']

print(f"\n✓ Best bandwidth: K = {best_bw}")


# 3. Compare logistic
X_const = sm.add_constant(X)
logit = sm.Logit(y, X_const).fit(disp=False)
logit_aicc = logit.aic + (2 * (logit.df_model + 1) * (logit.df_model + 2)) / (len(y) - logit.df_model - 2)
logit_auc = roc_auc_score(y.flatten(), logit.predict(X_const))

print("\n=== Model Comparison ===")
print(f"{'Metric':<10} {'Logistic':<15} {'GWLR':<15} {'Improvement':<15}")
print("-" * 55)
print(f"{'AICc':<10} {logit_aicc:<15.2f} {gwlr_aicc:<15.2f} {logit_aicc - gwlr_aicc:+.2f}")
print(f"{'AUC':<10} {logit_auc:<15.4f} {gwlr_auc:<15.4f} {gwlr_auc - logit_auc:+.4f}")

aicc_diff = logit_aicc - gwlr_aicc
print(f"\njudgment:")
if aicc_diff >= 4:
    print(f"  ✅ AICc reduce {aicc_diff:.1f} (≥4) → GWLR is significantly better than logistic")
elif aicc_diff >= 2:
    print(f"  ⚠️ AICc reduce {aicc_diff:.1f} (2-4) → Improved but not strong")
else:
    print(f"  ❌ AICc reduce {aicc_diff:.1f} (<2) → GWLR is equivalent to logistic")


# 4. Coefficient space statistics
local_params = gwlr_results.params
local_tvals = gwlr_results.tvalues

print("\n=== Coefficient Space Statistics ===")
print(f"{'Feature':<32} {'min':>8} {'max':>8} {'median':>8} {'IQR':>8} {'sig %':>8}")
print("-" * 80)
for i, feat in enumerate(FEATURES):
    col = local_params[:, i+1]
    tval = local_tvals[:, i+1]
    sig_pct = (np.abs(tval) > 1.96).mean() * 100
    print(f"{feat:<32} {col.min():>8.3f} {col.max():>8.3f} "
          f"{np.median(col):>8.3f} {np.percentile(col, 75) - np.percentile(col, 25):>8.3f} "
          f"{sig_pct:>7.1f}%")


# 5. Save coefficients + draw (same as above)
df_coef = df[['lon_wgs84', 'lat_wgs84', 'case']].copy()
df_coef['intercept'] = local_params[:, 0]
for i, feat in enumerate(FEATURES):
    df_coef[f'coef_{feat}'] = local_params[:, i+1]
    df_coef[f'tval_{feat}'] = local_tvals[:, i+1]
df_coef.to_csv("gwlr_local_coefficients.csv", index=False, encoding='utf-8-sig')
print(f"\n✓ Local coefficients saved to gwlr_local_coefficients.csv")

# 7 coefficient maps
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()
for i, feat in enumerate(FEATURES):
    ax = axes[i]
    col = local_params[:, i+1]
    tval = local_tvals[:, i+1]
    vmax = max(abs(col.min()), abs(col.max()))
    sig = np.abs(tval) > 1.96
    ax.scatter(df['lon_wgs84'][~sig], df['lat_wgs84'][~sig],
               c='lightgray', s=10, alpha=0.4, edgecolor='none')
    sc = ax.scatter(df['lon_wgs84'][sig], df['lat_wgs84'][sig],
                    c=col[sig], cmap='RdBu_r', s=25,
                    vmin=-vmax, vmax=vmax, edgecolor='none')
    ax.set_title(f"{feat}\n(sig: {sig.sum()}/{len(sig)})", fontsize=10)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(sc, ax=ax, shrink=0.7)
axes[-1].axis('off')
plt.suptitle(f"GWLR Local Coefficients (K={best_bw})", fontsize=12, y=1.00)
plt.tight_layout()
plt.savefig("gwlr_coefficient_maps.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Coefficient maps saved to gwlr_coefficient_maps.png")

# ROC comparison
fpr_l, tpr_l, _ = roc_curve(y.flatten(), logit.predict(X_const))
fpr_g, tpr_g, _ = roc_curve(y.flatten(), gwlr_results.predy.flatten())
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr_l, tpr_l, lw=2, label=f'Logistic (AUC={logit_auc:.3f})', color='steelblue')
ax.plot(fpr_g, tpr_g, lw=2, label=f'GWLR K={best_bw} (AUC={gwlr_auc:.3f})', color='crimson')
ax.plot([0,1], [0,1], 'k--', lw=1, alpha=0.5)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC: Logistic vs GWLR")
ax.legend(loc='lower right')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("roc_logistic_vs_gwlr.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ ROC comparison plot saved to roc_logistic_vs_gwlr.png")