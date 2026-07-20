import pandas as pd, numpy as np, warnings
import geopandas as gpd
import statsmodels.api as sm
from sklearn.cluster import KMeans
warnings.filterwarnings('ignore')

B_BOOT, SEED, UTM = 999, 42, "EPSG:32650"
FE = ['office_count','residential_count','mall_count','metro_station_count',
      'dist_to_nearest_luckin','dist_to_nearest_starbucks','dist_to_nearest_metro_exit']
DIST = [c for c in FE if c.startswith('dist_')]

def to_utm(df):
    gs = gpd.GeoSeries(gpd.points_from_xy(df['lon_wgs84'], df['lat_wgs84']),
                       crs="EPSG:4326").to_crs(UTM)
    return np.column_stack([gs.x.values, gs.y.values])

lk = pd.read_csv("task1_features_excl50_seed42.csv")
sb = pd.read_csv("task3_starbucks_features_excl50_seed42.csv")
lk['brand'], sb['brand'] = 1, 0                      # Luckin=1 (as specified in the handover document)
pool = pd.concat([lk, sb], ignore_index=True)
print(f"✓ Pooled {len(pool)} rows = {len(lk)} Luckin + {len(sb)} Starbucks")

# Distance variable: unified standardization of merged samples (ddof=0, consistent with the full text StandardScaler caliber)
pool[DIST] = (pool[DIST] - pool[DIST].mean()) / pool[DIST].std(ddof=0)

# Spatial block: k-means reproduces the 5 blocks of task 2 (same data and same seed), and Starbucks points are classified into the nearest block
km = KMeans(n_clusters=5, random_state=SEED, n_init=10).fit(to_utm(lk))
pool['block'] = np.r_[km.labels_, km.predict(to_utm(sb))]
print("block composition (row count | Luckin | Starbucks):")
for b in range(5):
    m = pool['block'] == b
    print(f"  block {b}: {m.sum()} | {pool.loc[m,'brand'].sum()} | {(m & (pool['brand']==0)).sum()}")

# Design matrix: 7 features + brand + brand×7
def design(df):
    X = df[FE].copy()
    X['brand'] = df['brand'].values
    for f in FE:
        X[f'brand_x_{f}'] = df['brand'].values * df[f].values
    return sm.add_constant(X, has_constant='add')

X_full, y_full = design(pool), pool['case'].astype(int)
main = sm.Logit(y_full, X_full).fit(disp=0, maxiter=200)
print(f"\n✓ Pooled model converged: {main.mle_retvals['converged']}, "
      f"McFadden R²={main.prsquared:.4f}")

# ----Cluster bootstrap: 5 blocks are redrawn with replacement ----
rng = np.random.default_rng(SEED)
boot, n_fail = [], 0
for b in range(B_BOOT):
    blocks = rng.integers(0, 5, size=5)
    df_b = pd.concat([pool[pool['block'] == g] for g in blocks], ignore_index=True)
    try:
        Xb = design(df_b)[X_full.columns]
        fit = sm.Logit(df_b['case'].astype(int), Xb).fit(disp=0, maxiter=200)
        if fit.mle_retvals['converged']:
            boot.append(fit.params[X_full.columns].values)
        else:
            n_fail += 1
    except Exception:
        n_fail += 1
boot = np.array(boot)
print(f"✓ bootstrap succeeded {len(boot)}/{B_BOOT} (failed {n_fail})")

ci_lo, ci_hi = np.percentile(boot, 2.5, axis=0), np.percentile(boot, 97.5, axis=0)
out = pd.DataFrame({'coef': main.params, 'OR': np.exp(main.params),
                    'p_asymptotic': main.pvalues,
                    'boot_CI_low': ci_lo, 'boot_CI_high': ci_hi,
                    'boot_sig': (ci_lo > 0) | (ci_hi < 0)}, index=X_full.columns)
out.to_csv("task3_pooled_model.csv", encoding='utf-8-sig')
print("\n=== Interaction terms (brand-specific effects, Luckin − Starbucks) ===")
print(out.loc[[c for c in out.index if c.startswith('brand_x_')]].round(4).to_string())

# ----Wald robustness (original method in Chapter 5: fitting by brand + respective standardization) ----
def fit_brand(df):
    X = df[FE].copy()
    X[DIST] = (X[DIST] - X[DIST].mean()) / X[DIST].std(ddof=0)
    m = sm.Logit(df['case'].astype(int), sm.add_constant(X)).fit(disp=0)
    return m.params, m.bse
p1, s1 = fit_brand(pd.read_csv("task1_features_excl50_seed42.csv"))
p0, s0 = fit_brand(pd.read_csv("task3_starbucks_features_excl50_seed42.csv"))
wald = pd.DataFrame({'luckin_coef': p1[FE], 'starbucks_coef': p0[FE],
                     'wald_Z': (p1[FE]-p0[FE])/np.sqrt(s1[FE]**2+s0[FE]**2)})
wald['wald_p'] = 2*(1-__import__('scipy.stats', fromlist=['norm']).norm.cdf(abs(wald['wald_Z'])))
wald.to_csv("task3_wald_robustness.csv", encoding='utf-8-sig')
print("\n=== Wald robustness===")
print(wald.round(4).to_string())
print("\nDirection consistency check: The variables of the interaction term boot_sig should roughly coincide with the variables of wald_p<0.05")