"""
Dissertation official version map is uniformly generated
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from matplotlib_scalebar.scalebar import ScaleBar
from shapely.geometry import Point
import contextily as cx
import warnings
warnings.filterwarnings('ignore')


# Global style configuration
plt.rcParams['font.family']    = 'DejaVu Sans'  # English academic font
plt.rcParams['font.size']      = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['figure.dpi']     = 100  # screen preview
SAVE_DPI = 300                         # Publication quality

WGS84 = "EPSG:4326"
WEB_MERC = "EPSG:3857"   # Used to overlay contextily basemaps


def add_north_arrow(ax, x=0.92, y=0.95):
    """Add a north arrow to the specified coordinates"""
    ax.annotate('N', xy=(x, y), xytext=(x, y - 0.08),
                xycoords='axes fraction',
                fontsize=12, ha='center', va='center',
                fontweight='bold',
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))


def add_scalebar(ax, location='lower right'):
    """Add scale. Web Mercator 1 unit ≈ 1 meter near the equator"""
    ax.add_artist(ScaleBar(1, location=location,
                           box_alpha=0.7, font_properties={'size': 9}))


def style_map_ax(ax, title=None):
    """Unified map style"""
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#333')
        spine.set_linewidth(0.8)


def add_basemap(ax, alpha=0.6):
    """Add OpenStreetMap basemap"""
    try:
        cx.add_basemap(ax, source=cx.providers.CartoDB.PositronNoLabels,
                       alpha=alpha)
    except Exception as e:
        print(f"  Warning: Failed to load basemap ({e}), skipping")


# Load all data
print("Loading data...")

# Administrative boundaries + sampling areas
siming = gpd.read_file("Siming_district.geojson").to_crs(WEB_MERC)
huli = gpd.read_file("Huli_district.geojson").to_crs(WEB_MERC)
sampling = gpd.read_file("sampling_area_final.geojson").to_crs(WEB_MERC)

# Luckin sample
features_lk = pd.read_csv("csv/features.csv")
gdf_cases_lk = gpd.GeoDataFrame(
    features_lk[features_lk['case'] == 1],
    geometry=gpd.points_from_xy(features_lk[features_lk['case'] == 1]['lon_wgs84'],
                                features_lk[features_lk['case'] == 1]['lat_wgs84']),
    crs=WGS84
).to_crs(WEB_MERC)
gdf_ctrl_lk = gpd.GeoDataFrame(
    features_lk[features_lk['case'] == 0],
    geometry=gpd.points_from_xy(features_lk[features_lk['case'] == 0]['lon_wgs84'],
                                features_lk[features_lk['case'] == 0]['lat_wgs84']),
    crs=WGS84
).to_crs(WEB_MERC)

# Starbucks sample
features_sb = pd.read_csv("csv/starbucks_features.csv")
gdf_cases_sb = gpd.GeoDataFrame(
    features_sb[features_sb['case'] == 1],
    geometry=gpd.points_from_xy(features_sb[features_sb['case'] == 1]['lon_wgs84'],
                                features_sb[features_sb['case'] == 1]['lat_wgs84']),
    crs=WGS84
).to_crs(WEB_MERC)

# LISA results
lisa = pd.read_csv("csv/moran_lisa_results.csv")
gdf_lisa = gpd.GeoDataFrame(
    lisa,
    geometry=gpd.points_from_xy(lisa['lon_wgs84'], lisa['lat_wgs84']),
    crs=WGS84
).to_crs(WEB_MERC)

# GWLR coefficient
gwlr = pd.read_csv("csv/gwlr_local_coefficients.csv")
gdf_gwlr = gpd.GeoDataFrame(
    gwlr,
    geometry=gpd.points_from_xy(gwlr['lon_wgs84'], gwlr['lat_wgs84']),
    crs=WGS84
).to_crs(WEB_MERC)

print("✓ Data loading complete")


# Figure 1: Study Area
print("\nGenerating Figure 1: Study Area...")

fig, ax = plt.subplots(figsize=(10, 10))

# Siming + Huli District
siming.plot(ax=ax, facecolor='#FFE5B4', edgecolor='#333', lw=1.5, alpha=0.6)
huli.plot(ax=ax, facecolor='#B4D8E5', edgecolor='#333', lw=1.5, alpha=0.6)

add_basemap(ax, alpha=0.5)

# Administrative district labeling
for gdf_dist, name, color in [(siming, 'Siming\nDistrict', '#8B4513'),
                                (huli, 'Huli\nDistrict', '#1F4E79')]:
    centroid = gdf_dist.geometry.unary_union.centroid
    ax.annotate(name, xy=(centroid.x, centroid.y),
                ha='center', va='center',
                fontsize=13, fontweight='bold', color=color,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='none', alpha=0.7))

# legend
legend_patches = [
    mpatches.Patch(facecolor='#FFE5B4', edgecolor='#333', label='Siming District'),
    mpatches.Patch(facecolor='#B4D8E5', edgecolor='#333', label='Huli District'),
]
ax.legend(handles=legend_patches, loc='upper left', frameon=True, fontsize=10)

style_map_ax(ax, "Study Area: Xiamen Island")
add_north_arrow(ax)
add_scalebar(ax)

plt.tight_layout()
plt.savefig("png/fig01_study_area.png", dpi=SAVE_DPI, bbox_inches='tight')
plt.close()
print("  ✓ png/fig01_study_area.png")


# Figure 3: Sampling Design
print("\nGenerating Figure 3: Sampling Design...")

fig, ax = plt.subplots(figsize=(11, 10))

sampling.plot(ax=ax, facecolor='#C8E6C9', edgecolor='none', alpha=0.55)
siming.boundary.plot(ax=ax, color='#666', lw=0.8, ls='--')
huli.boundary.plot(ax=ax, color='#666', lw=0.8, ls='--')

add_basemap(ax, alpha=0.5)

# Draw control first (small blue dot), then draw case (big red dot)
gdf_ctrl_lk.plot(ax=ax, color='#3F51B5', markersize=6, alpha=0.4, label=f'Controls (n={len(gdf_ctrl_lk)})')
gdf_cases_lk.plot(ax=ax, color='#D32F2F', markersize=22, alpha=0.85,
                  edgecolor='white', lw=0.4, label=f'Luckin cases (n={len(gdf_cases_lk)})')

# Custom legend
legend_elements = [
    mpatches.Patch(facecolor='#C8E6C9', edgecolor='none', alpha=0.55, label='Sampling area'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#D32F2F',
               markersize=9, label=f'Luckin cases (n={len(gdf_cases_lk)})'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#3F51B5',
               markersize=6, alpha=0.6, label=f'Controls (n={len(gdf_ctrl_lk)})'),
]
ax.legend(handles=legend_elements, loc='upper left', frameon=True, fontsize=10)

style_map_ax(ax, "Case-Control Sampling Design (1:5 ratio)")
add_north_arrow(ax)
add_scalebar(ax)

plt.tight_layout()
plt.savefig("png/fig03_sampling_design.png", dpi=SAVE_DPI, bbox_inches='tight')
plt.close()
print("  ✓ png/fig03_sampling_design.png")


# Figure 4: Bivariate Distribution (Boxplots)
print("\nGenerating Figure 4: Bivariate Distributions...")

feature_cols = ['office_count', 'residential_count', 'mall_count', 'metro_station_count',
                'dist_to_nearest_luckin', 'dist_to_nearest_starbucks', 'dist_to_nearest_metro_exit']
labels = ['Office\ncount', 'Residential\ncount', 'Mall\ncount', 'Metro station\ncount',
          'Dist. to\nLuckin (m)', 'Dist. to\nStarbucks (m)', 'Dist. to\nMetro exit (m)']

fig, axes = plt.subplots(2, 4, figsize=(15, 8))
axes = axes.flatten()

for i, (col, lab) in enumerate(zip(feature_cols, labels)):
    ax = axes[i]
    data = [features_lk[features_lk['case'] == 0][col],
            features_lk[features_lk['case'] == 1][col]]
    bp = ax.boxplot(data, tick_labels=['Control', 'Case'],
                    patch_artist=True, showfliers=False, widths=0.6)
    bp['boxes'][0].set_facecolor('#3F51B5')
    bp['boxes'][0].set_alpha(0.6)
    bp['boxes'][1].set_facecolor('#D32F2F')
    bp['boxes'][1].set_alpha(0.6)
    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(1.5)
    ax.set_title(lab, fontsize=10)
    ax.grid(axis='y', alpha=0.3, ls='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

axes[-1].axis('off')
plt.suptitle("Bivariate Distribution of Features: Cases vs Controls",
             fontsize=12, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig("png/fig04_bivariate_distributions.png", dpi=SAVE_DPI, bbox_inches='tight')
plt.close()
print("  ✓ png/fig04_bivariate_distributions.png")


# Figure 6: LISA Cluster Map
print("\nGenerating Figure 6: LISA Cluster Map...")

fig, ax = plt.subplots(figsize=(11, 10))

# Background: Siming + Huli light background
sampling.plot(ax=ax, facecolor='#F5F5F5', edgecolor='none', alpha=0.5)
siming.boundary.plot(ax=ax, color='#888', lw=0.6, ls='--')
huli.boundary.plot(ax=ax, color='#888', lw=0.6, ls='--')

add_basemap(ax, alpha=0.5)

# Draw ns first, then draw the salient ones (make sure the salient points are not blocked)
color_map = {
    'ns': ('#CCCCCC', 4, 0.3),
    'HH': ('#D32F2F', 35, 0.9),
    'LL': ('#1976D2', 35, 0.9),
    'HL': ('#FFC107', 28, 0.85),
    'LH': ('#7E57C2', 28, 0.85),
}
order = ['ns', 'LH', 'HL', 'LL', 'HH']

legend_elements = []
for label in order:
    sub = gdf_lisa[gdf_lisa['lisa_label'] == label]
    if len(sub) == 0:
        continue
    color, size, alpha = color_map[label]
    sub.plot(ax=ax, color=color, markersize=size, alpha=alpha,
             edgecolor='none' if label == 'ns' else 'white', lw=0.3)
    if label != 'ns':
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                         markerfacecolor=color, markersize=10,
                                         label=f"{label} (n={len(sub)})"))

legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor='#CCCCCC', markersize=6,
                                  label=f"Not significant (n={len(gdf_lisa[gdf_lisa['lisa_label']=='ns'])})"))

ax.legend(handles=legend_elements, loc='upper left', frameon=True, fontsize=10,
          title='LISA Cluster Type', title_fontsize=10)

style_map_ax(ax, "LISA Cluster Map of Logistic Regression Residuals\n"
                 "(HH = high residual cluster; LL = low residual cluster; HL/LH = spatial outliers)")
add_north_arrow(ax)
add_scalebar(ax)

plt.tight_layout()
plt.savefig("png/fig06_lisa_cluster_map.png", dpi=SAVE_DPI, bbox_inches='tight')
plt.close()
print("  ✓ png/fig06_lisa_cluster_map.png")


# Figure 7: GWLR Coefficient Maps (6 sub-panels)
print("\nGenerating Figure 7: GWLR Coefficient Maps...")

GWLR_FEATURES = [
    ('coef_office_count', 'tval_office_count', 'Office count'),
    ('coef_residential_count', 'tval_residential_count', 'Residential count'),
    ('coef_mall_count', 'tval_mall_count', 'Mall count'),
    ('coef_dist_to_nearest_luckin', 'tval_dist_to_nearest_luckin', 'Distance to nearest Luckin'),
    ('coef_dist_to_nearest_starbucks', 'tval_dist_to_nearest_starbucks', 'Distance to nearest Starbucks'),
    ('coef_dist_to_nearest_metro_exit', 'tval_dist_to_nearest_metro_exit', 'Distance to nearest metro exit'),
]

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i, (coef_col, tval_col, title) in enumerate(GWLR_FEATURES):
    ax = axes[i]

    sampling.plot(ax=ax, facecolor='#F5F5F5', edgecolor='none', alpha=0.4)
    siming.boundary.plot(ax=ax, color='#888', lw=0.5, ls='--')
    huli.boundary.plot(ax=ax, color='#888', lw=0.5, ls='--')
    add_basemap(ax, alpha=0.4)

    coef = gdf_gwlr[coef_col].values
    tval = gdf_gwlr[tval_col].values
    sig = np.abs(tval) > 1.96

    vmax = max(abs(coef.min()), abs(coef.max()))

    # Inconspicuous point: light gray
    gdf_gwlr[~sig].plot(ax=ax, color='lightgray', markersize=4, alpha=0.3)
    # Notable points: divergent color matching
    if sig.sum() > 0:
        sc = ax.scatter(gdf_gwlr[sig].geometry.x, gdf_gwlr[sig].geometry.y,
                        c=coef[sig], cmap='RdBu_r', s=18,
                        vmin=-vmax, vmax=vmax, edgecolor='none', alpha=0.85)
        cbar = plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
        cbar.set_label('Coefficient (β)', fontsize=8)
        cbar.ax.tick_params(labelsize=8)

    style_map_ax(ax, f"{title}\n(significant n={sig.sum()}/{len(sig)})")

plt.suptitle("GWLR Local Coefficient Maps (bandwidth K=400)\n"
             "Red = positive local effect; Blue = negative local effect; Gray = not statistically significant",
             fontsize=12, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig("png/fig07_gwlr_coefficient_maps.png", dpi=SAVE_DPI, bbox_inches='tight')
plt.close()
print("  ✓ png/fig07_gwlr_coefficient_maps.png")


# Figure 11: ROC Comparison (Logistic vs GWLR vs XGBoost-CV)
print("\nGenerating Figure 11: ROC Comparison...")

import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb

# Rerun the model to get clean ROC data
FEATURES = ['office_count', 'residential_count', 'mall_count', 'metro_station_count',
            'dist_to_nearest_luckin', 'dist_to_nearest_starbucks', 'dist_to_nearest_metro_exit']
X = features_lk[FEATURES].copy()
dist_cols = [c for c in FEATURES if c.startswith('dist_')]
X[dist_cols] = StandardScaler().fit_transform(X[dist_cols])
y = features_lk['case'].astype(int).values

# Logistic
X_const = sm.add_constant(X)
logit = sm.Logit(y, X_const).fit(disp=False)
logit_proba = logit.predict(X_const).values
logit_auc = roc_auc_score(y, logit_proba)
fpr_l, tpr_l, _ = roc_curve(y, logit_proba)

# XGBoost — Use CV prediction to avoid the overfitting artifact of training AUC=1
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
xgb_proba = np.zeros(len(y))
for tr, te in cv.split(X, y):
    m = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, scale_pos_weight=5,
                          eval_metric='auc', random_state=42, use_label_encoder=False)
    m.fit(X.iloc[tr], y[tr])
    xgb_proba[te] = m.predict_proba(X.iloc[te])[:, 1]
xgb_auc = roc_auc_score(y, xgb_proba)
fpr_x, tpr_x, _ = roc_curve(y, xgb_proba)

# GWLR: Reconstruct predicted probabilities from existing gwlr_local_coefficients.csv
# Calculate the log-odds of each point using local β and then convert the probability
gwlr_local = pd.read_csv("csv/gwlr_local_coefficients.csv")
gwlr_coefs = ['intercept'] + [f'coef_{f}' for f in FEATURES if f != 'metro_station_count']
gwlr_xfeats = [f for f in FEATURES if f != 'metro_station_count']
X_gwlr = features_lk[gwlr_xfeats].copy()
X_gwlr[dist_cols] = StandardScaler().fit_transform(X_gwlr[dist_cols])
linear = gwlr_local['intercept'].values + \
         sum(gwlr_local[f'coef_{f}'].values * X_gwlr[f].values for f in gwlr_xfeats)
gwlr_proba = 1 / (1 + np.exp(-linear))
gwlr_auc = roc_auc_score(y, gwlr_proba)
fpr_g, tpr_g, _ = roc_curve(y, gwlr_proba)

# Draw a picture
fig, ax = plt.subplots(figsize=(8, 7))
ax.plot(fpr_l, tpr_l, color='#1976D2', lw=2.2, label=f'Logistic (AUC = {logit_auc:.3f})')
ax.plot(fpr_g, tpr_g, color='#388E3C', lw=2.2, label=f'GWLR (AUC = {gwlr_auc:.3f})')
ax.plot(fpr_x, tpr_x, color='#D32F2F', lw=2.2, label=f'XGBoost 5-fold CV (AUC = {xgb_auc:.3f})')
ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random classifier')
ax.set_xlabel("False Positive Rate", fontsize=11)
ax.set_ylabel("True Positive Rate", fontsize=11)
ax.set_title("ROC Curves: Three Modelling Approaches", fontsize=12, fontweight='bold')
ax.legend(loc='lower right', frameon=True, fontsize=10)
ax.grid(alpha=0.3)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.01)

plt.tight_layout()
plt.savefig("png/fig11_roc_comparison.png", dpi=SAVE_DPI, bbox_inches='tight')
plt.close()
print(f"  ✓ png/fig11_roc_comparison.png  (Logistic={logit_auc:.3f}, GWLR={gwlr_auc:.3f}, XGBoost-CV={xgb_auc:.3f})")


# Figure 13: Luckin vs Starbucks Spatial Distribution
print("\nGenerating Figure 13: Luckin vs Starbucks Spatial...")

fig, ax = plt.subplots(figsize=(11, 10))

sampling.plot(ax=ax, facecolor='#F5F5F5', edgecolor='none', alpha=0.4)
siming.boundary.plot(ax=ax, color='#888', lw=0.6, ls='--')
huli.boundary.plot(ax=ax, color='#888', lw=0.6, ls='--')
add_basemap(ax, alpha=0.5)

gdf_cases_lk.plot(ax=ax, color='#D32F2F', markersize=35, alpha=0.7,
                  edgecolor='white', lw=0.4, marker='o',
                  label=f'Luckin (n={len(gdf_cases_lk)})')
gdf_cases_sb.plot(ax=ax, color='#2E7D32', markersize=50, alpha=0.8,
                  edgecolor='white', lw=0.4, marker='^',
                  label=f'Starbucks (n={len(gdf_cases_sb)})')

ax.legend(loc='upper left', frameon=True, fontsize=10, title='Brand',
          title_fontsize=10, markerscale=1.0)

# style_map_ax(ax, "Spatial Distribution of Luckin and Starbucks Stores on Xiamen Island")
add_north_arrow(ax)
add_scalebar(ax)

plt.tight_layout()
plt.savefig("png/fig13_luckin_vs_starbucks_spatial.png", dpi=SAVE_DPI, bbox_inches='tight')
plt.close()
print("  ✓ png/fig13_luckin_vs_starbucks_spatial.png")


print("\n" + "="*60)
print("✅ All dissertation figures have been generated successfully")
print("="*60)