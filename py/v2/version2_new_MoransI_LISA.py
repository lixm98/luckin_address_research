import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as cx

from matplotlib_scalebar.scalebar import ScaleBar
from libpysal.weights import KNN, DistanceBand
from esda.moran import Moran, Moran_Local
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")
np.random.seed(42)

# Global mapping settings
WGS84 = "EPSG:4326"
UTM50N = "EPSG:32650"
WEB_MERC = "EPSG:3857"
SAVE_DPI = 300

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 11
plt.rcParams["axes.labelsize"] = 10


def add_north_arrow(ax, x=0.92, y=0.95):
    """Add a north arrow."""
    ax.annotate(
        "N",
        xy=(x, y),
        xytext=(x, y - 0.08),
        xycoords="axes fraction",
        fontsize=12,
        ha="center",
        va="center",
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", lw=1.5, color="black"),
    )


def add_scalebar(ax, location="lower right"):
    """Add a scale bar on the EPSG:3857 layer."""
    ax.add_artist(
        ScaleBar(
            1,
            location=location,
            box_alpha=0.7,
            font_properties={"size": 9},
        )
    )


def style_map_ax(ax, title=None):
    """Unify the map axis style."""
    ax.set_xticks([])
    ax.set_yticks([])

    if title:
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)

    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")
        spine.set_linewidth(0.8)


def add_basemap(ax, alpha=0.5):
    """Add CartoDB light basemap consistent with Final_visualization."""
    try:
        cx.add_basemap(
            ax,
            source=cx.providers.CartoDB.PositronNoLabels,
            alpha=alpha,
        )
    except Exception as e:
        print(f"Warning: Unable to load basemap, skipped. reason:{e}")


# Read residuals and coordinates
print("Loading residual and boundary data...")

df = pd.read_csv("csv/residuals_excel50.csv")

assert len(df) == 1056, (
    f"Row count={len(df)}，should be 1056。"
    "Please confirm that residuals_excel50.csv is the final Pearson residual file after re-running the 50m data。"
)

required_cols = {"lon_wgs84", "lat_wgs84", "pearson_residual"}
missing_cols = required_cols - set(df.columns)

if missing_cols:
    raise ValueError(f"csv/residuals_excel50.csv Required columns are missing:{missing_cols}")

# Moran/LISA weight and distance calculations must use metric projection
gs_utm = gpd.GeoSeries(
    gpd.points_from_xy(df["lon_wgs84"], df["lat_wgs84"]),
    crs=WGS84,
).to_crs(UTM50N)

coords = np.column_stack([gs_utm.x.to_numpy(), gs_utm.y.to_numpy()])
resid = df["pearson_residual"].to_numpy()

# For maps: Go to Web Mercator, overlay with contextily basemap
gdf_lisa = gpd.GeoDataFrame(
    df.copy(),
    geometry=gpd.points_from_xy(df["lon_wgs84"], df["lat_wgs84"]),
    crs=WGS84,
).to_crs(WEB_MERC)

# Study area/administrative area layer consistent with older version of Figure 6
sampling = gpd.read_file("geojson/sampling_area_final.geojson").to_crs(WEB_MERC)
siming = gpd.read_file("geojson/Siming_district.geojson").to_crs(WEB_MERC)
huli = gpd.read_file("geojson/Huli_district.geojson").to_crs(WEB_MERC)

print("✓ Data loading completed")

# 1) Global Moran's I: 6 weight setting sensitivity grids
settings = (
    [("KNN", k) for k in (6, 8, 10)]
    + [("DistBand", d) for d in (800, 1000, 1200)]
)

rows = []

print("\n=== Global Moran's I（999 permutations, row-standardized weights）===")

for kind, param in settings:
    if kind == "KNN":
        w = KNN.from_array(coords, k=param)
    else:
        w = DistanceBand.from_array(
            coords,
            threshold=param,
            binary=True,
            silence_warnings=False,
        )

    w.transform = "r"
    moran = Moran(resid, w, permutations=999)

    rows.append(
        {
            "weight": kind,
            "param": param,
            "mean_neighbors": round(w.mean_neighbors, 1),
            "I": round(moran.I, 4),
            "z": round(moran.z_sim, 3),
            "p": round(moran.p_sim, 4),
        }
    )

    print(
        f"{kind:<9}{param:>5}: "
        f"I={moran.I:+.4f}, z={moran.z_sim:+.3f}, p={moran.p_sim:.4f}"
    )

grid = pd.DataFrame(rows)
grid.to_csv("csv/task4_moran_grid.csv", index=False, encoding="utf-8-sig")

if (grid["p"] > 0.05).all():
    print("Consistency: All non-significant (p > 0.05)")
else:
    print("Consistency: Significant settings exist — please check each row")

# 2) LISA: Main setting KNN k=8, 999 permutations, BH-FDR
w8 = KNN.from_array(coords, k=8)
w8.transform = "r"

lisa = Moran_Local(resid, w8, permutations=999)

sig_raw = lisa.p_sim < 0.05
sig_fdr = multipletests(
    lisa.p_sim,
    alpha=0.05,
    method="fdr_bh",
)[0]


def classify_lisa(significant, quadrants):
    """Map LISA quadrants and significance to HH/LH/LL/HL/ns."""
    labels = np.full(len(quadrants), "ns", dtype=object)

    quadrant_map = {
        1: "HH",
        2: "LH",
        3: "LL",
        4: "HL",
    }

    for code, label in quadrant_map.items():
        labels[significant & (quadrants == code)] = label

    return labels


lab_raw = classify_lisa(sig_raw, lisa.q)
lab_fdr = classify_lisa(sig_fdr, lisa.q)

print("\n=== LISA Classification: Uncorrected vs FDR(BH) Corrected ===")
print(f"{'category':<6}{'uncalibrated':>10}{'After FDR correction':>12}")

for category in ["HH", "LL", "LH", "HL", "ns"]:
    raw_n = (lab_raw == category).sum()
    fdr_n = (lab_fdr == category).sum()
    print(f"{category:<6}{raw_n:>10}{fdr_n:>12}")

print(f"Total significant points:{sig_raw.sum()} → {sig_fdr.sum()}")

df["lisa_raw"] = lab_raw
df["lisa_fdr"] = lab_fdr
df["lisa_p_sim"] = lisa.p_sim
df["lisa_q"] = lisa.q

df.to_csv(
    "csv/task4_lisa_results.csv",
    index=False,
    encoding="utf-8-sig",
)

# Update the cartographic GeoDataFrame to have LISA labels
gdf_lisa["lisa_raw"] = lab_raw
gdf_lisa["lisa_fdr"] = lab_fdr
gdf_lisa["lisa_p_sim"] = lisa.p_sim
gdf_lisa["lisa_q"] = lisa.q

# 3) FDR corrected LISA map: base map, boundary, scale bar, north arrow
print("\nGenerating FDR-corrected LISA map...")

fig, ax = plt.subplots(figsize=(11, 10))

# The light gray surface of the study area and the dotted boundary of the administrative area
sampling.plot(
    ax=ax,
    facecolor="#F5F5F5",
    edgecolor="none",
    alpha=0.5,
    zorder=1,
)

siming.boundary.plot(
    ax=ax,
    color="#888888",
    linewidth=0.6,
    linestyle="--",
    zorder=2,
)

huli.boundary.plot(
    ax=ax,
    color="#888888",
    linewidth=0.6,
    linestyle="--",
    zorder=2,
)

# Base plot should be drawn before scatter points
add_basemap(ax, alpha=0.5)

# Classification colors and drawing order consistent with the old pictures
colour_map = {
    "ns": ("#E14D4DFF", 4, 0.30),
    "HH": ("#575656", 35, 0.90),
    "LL": ("#1976D2", 35, 0.90),
    "HL": ("#FFC107", 28, 0.85),
    "LH": ("#7E57C2", 28, 0.85),
}

plot_order = ["ns", "LH", "HL", "LL", "HH"]
legend_elements = []

for label in plot_order:
    subset = gdf_lisa[gdf_lisa["lisa_fdr"] == label]

    if subset.empty:
        continue

    colour, size, alpha = colour_map[label]

    subset.plot(
        ax=ax,
        color=colour,
        markersize=size,
        alpha=alpha,
        edgecolor="none" if label == "ns" else "white",
        linewidth=0.3,
        zorder=3 if label == "ns" else 4,
    )

    if label != "ns":
        legend_elements.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=colour,
                markeredgecolor="white",
                markersize=10,
                label=f"{label} (n={len(subset)})",
            )
        )

# The ns legend is shown regardless of whether there are significant points after FDR.
n_ns = (gdf_lisa["lisa_fdr"] == "ns").sum()

legend_elements.append(
    plt.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="#E14D4DFF",
        markeredgecolor="none",
        markersize=6,
        label=f"Not significant after FDR (n={n_ns})",
    )
)

ax.legend(
    handles=legend_elements,
    loc="upper left",
    frameon=True,
    fontsize=10,
    title="LISA Cluster Type",
    title_fontsize=10,
)

style_map_ax(
    ax,
    "LISA Clusters of Logistic Regression Residuals\n"
    "(KNN k=8; BH FDR-corrected; 999 permutations)",
)

add_north_arrow(ax)
add_scalebar(ax)

plt.tight_layout()

plt.savefig(
    "png/task4_lisa_map_fdr.png",
    dpi=SAVE_DPI,
    bbox_inches="tight",
)

plt.close()

print("✓ task4_lisa_map_fdr.png(Including base map, scale bar, compass)")

# 4) Method record
print("\n=== Appendix A / §3.4.2 Record ===")
print("Residual: Pearson residual = (y - p_hat) / sqrt[p_hat(1 - p_hat)], from the global logistic。")
print("Weighted coordinate projection: EPSG:32650; weight: row normalized (r); number of permutations: 999; random seed: 42.")
print("LISA main settings: KNN k=8; multiple comparison correction: Benjamini-Hochberg FDR, alpha=0.05.")
print("\n✓ Output:")
print("  - csv/task4_moran_grid.csv")
print("  - csv/task4_lisa_results.csv")
print("  - png/task4_lisa_map_fdr.png")