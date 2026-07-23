# Moran scatter plot - Pearson residuals of the main logistic model
# Input: residuals_excel50.csv (lon_wgs84, lat_wgs84, pearson_residual)
# Weights: KNN k=8 on EPSG:32650 coordinates, row-standardised (matches version2_MoransI_LISA.py)
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from pyproj import Transformer

r = pd.read_csv('csv/residuals_excel50.csv')
tr = Transformer.from_crs("EPSG:4326", "EPSG:32650", always_xy=True)
x, y = tr.transform(r.lon_wgs84.values, r.lat_wgs84.values)
XY = np.c_[x, y]
D = np.sqrt(((XY[:, None, :] - XY[None, :, :]) ** 2).sum(-1))
np.fill_diagonal(D, np.inf)
idx = np.argsort(D, axis=1)[:, :8]
n = len(r)
W = np.zeros((n, n))
for i in range(n):
    W[i, idx[i]] = 1 / 8
z = ((r.pearson_residual - r.pearson_residual.mean()) / r.pearson_residual.std(ddof=0)).values
lag = W @ z
I = (z @ lag) / (z @ z)
print("Moran's I =", round(I, 4))

fig, ax = plt.subplots(figsize=(8, 7))
ax.scatter(z, lag, s=14, alpha=0.45, color='steelblue', edgecolors='none')
xs = np.linspace(z.min(), z.max(), 100)
ax.plot(xs, I * xs, color='crimson', lw=2, label=f"Slope = Moran's I = {I:.4f}")
ax.axhline(0, color='grey', lw=0.8, ls='--')
ax.axvline(0, color='grey', lw=0.8, ls='--')
ax.set_xlabel('Standardised Pearson residual')
ax.set_ylabel('Spatial lag of residual (KNN, k = 8)')
ax.set_title('Moran Scatter Plot \u2013 Pearson Residuals (KNN k=8)')
ax.legend(loc='lower right')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('png/moran_scatter_excl50.png', dpi=150)
