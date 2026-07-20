"""
Moran's I sensitivity grid (6 settings) + LISA 999 replacement + FDR (BH) correction
"""
import pandas as pd, numpy as np, warnings
import geopandas as gpd
from libpysal.weights import KNN, DistanceBand
from esda.moran import Moran, Moran_Local
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

np.random.seed(42)   # Fixed permutation random source

df = pd.read_csv("residuals_excel50.csv")
assert len(df) == 1056, f"rows={len(df)}, should be 1056——confirm it's the re-run residuals.csv for 50m data"
gs = gpd.GeoSeries(gpd.points_from_xy(df['lon_wgs84'], df['lat_wgs84']),
                   crs="EPSG:4326").to_crs("EPSG:32650")
coords = np.column_stack([gs.x.values, gs.y.values])
resid = df['pearson_residual'].values

# ----1) Global Moran's I: 6 Set sensitivity grid ----
settings = [("KNN", k) for k in (6, 8, 10)] + [("DistBand", d) for d in (800, 1000, 1200)]
rows = []
print("=== Global Moran's I (999 permutations, row-standardized weights) ===")
for kind, param in settings:
    w = KNN.from_array(coords, k=param) if kind == "KNN" \
        else DistanceBand.from_array(coords, threshold=param, binary=True)
    w.transform = 'r'
    m = Moran(resid, w, permutations=999)
    rows.append(dict(weight=kind, param=param, mean_neighbors=round(w.mean_neighbors, 1),
                     I=round(m.I, 4), z=round(m.z_sim, 3), p=round(m.p_sim, 4)))
    print(f"  {kind:<9}{param:>5}: I={m.I:+.4f}, z={m.z_sim:+.3f}, p={m.p_sim:.4f}")
grid = pd.DataFrame(rows)
grid.to_csv("task4_moran_grid.csv", index=False, encoding="utf-8-sig")
print(f"Consistency: {'All non-significant (p>0.05)' if (grid['p'] > 0.05).all() else 'Significant settings exist — check row by row'}")

# ----2) LISA (main setting KNN k=8) + FDR correction ----
w8 = KNN.from_array(coords, k=8); w8.transform = 'r'
lisa = Moran_Local(resid, w8, permutations=999)
sig_raw = lisa.p_sim < 0.05
sig_fdr = multipletests(lisa.p_sim, alpha=0.05, method='fdr_bh')[0]   # Benjamin Hochberg

def classify(sig):
    lab = np.array(['ns'] * len(resid), dtype=object)
    for code, name in [(1,'HH'), (2,'LH'), (3,'LL'), (4,'HL')]:
        lab[sig & (lisa.q == code)] = name
    return lab
lab_raw, lab_fdr = classify(sig_raw), classify(sig_fdr)

print("\n=== LISA Classification: Uncorrected vs FDR(BH) Corrected ===")
print(f"{'Category':<6}{'Uncorrected':>8}{'FDR Corrected':>10}")
for c in ['HH', 'LL', 'LH', 'HL', 'ns']:
    print(f"{c:<6}{(lab_raw==c).sum():>8}{(lab_fdr==c).sum():>10}")
print(f"Significant points total: {sig_raw.sum()} → {sig_fdr.sum()}")

df['lisa_raw'], df['lisa_fdr'] = lab_raw, lab_fdr
df['lisa_p_sim'] = lisa.p_sim
df.to_csv("task4_lisa_results.csv", index=False, encoding="utf-8-sig")

# ----3) LISA map after FDR correction ----
cmap = {'HH':'red', 'LL':'blue', 'HL':'pink', 'LH':'lightblue', 'ns':'gray'}
fig, ax = plt.subplots(figsize=(11, 10))
for lab, color in cmap.items():
    sub = df[df['lisa_fdr'] == lab]
    if len(sub) == 0: continue
    ax.scatter(sub['lon_wgs84'], sub['lat_wgs84'], c=color,
               s=35 if lab != 'ns' else 8, alpha=0.85 if lab != 'ns' else 0.3,
               label=f"{lab} (n={len(sub)})")
ax.set_title("LISA clusters, FDR-corrected (KNN k=8, 999 permutations)")
ax.legend(loc='upper left'); ax.set_aspect('equal')
plt.tight_layout()
plt.savefig("task4_lisa_map_fdr.png", dpi=150, bbox_inches='tight')
plt.close()

# ----4) Appendix A records ----
print("\n=== Appendix A / §3.4.2 Records ===")
print("Residuals: Pearson residual (y - p_hat)/sqrt(p_hat(1-p_hat)), from global logistic")
print("Projection: EPSG:32650 | Weights: row-standardized('r') | Permutations: 999 | Global seed: numpy seed=42")
print("LISA main setting: KNN k=8 | Multiple correction: Benjamini-Hochberg FDR, alpha=0.05")
print("✓ task4_moran_grid.csv / task4_lisa_results.csv / task4_lisa_map_fdr.png")