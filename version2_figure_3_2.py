import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib_scalebar.scalebar import ScaleBar
import contextily as cx

# parameter
SAVE_DPI = 300
WGS84 = "EPSG:4326"
WEB_MERC = "EPSG:3857"

# compass
def add_north_arrow(ax, x=0.92, y=0.95):
    ax.annotate(
        'N',
        xy=(x, y),
        xytext=(x, y-0.08),
        xycoords='axes fraction',
        fontsize=12,
        ha='center',
        va='center',
        fontweight='bold',
        arrowprops=dict(arrowstyle='->', lw=1.5, color='black')
    )

# scale
def add_scalebar(ax):
    ax.add_artist(
        ScaleBar(
            1,
            location='lower right',
            box_alpha=0.7,
            font_properties={'size':9}
        )
    )

# map style
def style_map_ax(ax, title):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=11, fontweight='bold')

    for spine in ax.spines.values():
        spine.set_edgecolor('#333')
        spine.set_linewidth(0.8)

# base map
def add_basemap(ax):
    cx.add_basemap(
        ax,
        source=cx.providers.CartoDB.PositronNoLabels,
        alpha=0.5
    )

# Read data

sampling = gpd.read_file(
    "sampling_area_final.geojson"
).to_crs(WEB_MERC)

siming = gpd.read_file(
    "Siming_district.geojson"
).to_crs(WEB_MERC)

huli = gpd.read_file(
    "Huli_district.geojson"
).to_crs(WEB_MERC)

df = pd.read_csv("task1_features_excl50_seed42.csv")

# Convert GeoDataFrame

gdf_cases = gpd.GeoDataFrame(
    df[df["case"] == 1],
    geometry=gpd.points_from_xy(
        df[df["case"] == 1]["lon_wgs84"],
        df[df["case"] == 1]["lat_wgs84"]
    ),
    crs=WGS84
).to_crs(WEB_MERC)

gdf_controls = gpd.GeoDataFrame(
    df[df["case"] == 0],
    geometry=gpd.points_from_xy(
        df[df["case"] == 0]["lon_wgs84"],
        df[df["case"] == 0]["lat_wgs84"]
    ),
    crs=WGS84
).to_crs(WEB_MERC)

# Start drawing

fig, ax = plt.subplots(figsize=(11,10))

sampling.plot(
    ax=ax,
    facecolor="#C8E6C9",
    edgecolor="none",
    alpha=0.55
)

siming.boundary.plot(
    ax=ax,
    color="#666",
    lw=0.8,
    ls="--"
)

huli.boundary.plot(
    ax=ax,
    color="#666",
    lw=0.8,
    ls="--"
)

# base map
add_basemap(ax)

# Controls
gdf_controls.plot(
    ax=ax,
    color="#3F51B5",
    markersize=6,
    alpha=0.4,
    label=f"Controls (n={len(gdf_controls)})"
)

# Luckin
gdf_cases.plot(
    ax=ax,
    color="#D32F2F",
    markersize=22,
    alpha=0.85,
    edgecolor="white",
    linewidth=0.4,
    label=f"Luckin cases (n={len(gdf_cases)})"
)

# legend
legend_elements = [
    mpatches.Patch(
        facecolor="#C8E6C9",
        edgecolor="none",
        alpha=0.55,
        label="Sampling area"
    ),
    plt.Line2D(
        [0],[0],
        marker='o',
        color='w',
        markerfacecolor='#D32F2F',
        markersize=9,
        label=f'Luckin cases (n={len(gdf_cases)})'
    ),
    plt.Line2D(
        [0],[0],
        marker='o',
        color='w',
        markerfacecolor='#3F51B5',
        markersize=6,
        alpha=0.6,
        label=f'Controls (n={len(gdf_controls)})'
    )
]

ax.legend(
    handles=legend_elements,
    loc="upper left",
    frameon=True,
    fontsize=10
)

style_map_ax(
    ax,
    "Case-Control Sampling Design (1:5 ratio)"
)

add_north_arrow(ax)
add_scalebar(ax)

plt.tight_layout()

plt.savefig(
    "figure_3_2_sample_excl50_seed42.png",
    dpi=SAVE_DPI,
    bbox_inches="tight"
)

plt.show()