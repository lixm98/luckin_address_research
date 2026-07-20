import pandas as pd, numpy as np
import geopandas as gpd

lk = pd.read_csv("xiamen_luckin_poi_wgs84.csv")
gs = gpd.GeoSeries(gpd.points_from_xy(lk['lon_wgs84'], lk['lat_wgs84']),
                   crs="EPSG:4326").to_crs("EPSG:32650")
xy = np.column_stack([gs.x, gs.y])

island = lk['adname'].isin(['思明区', '湖里区'])
dup_names = lk.loc[lk['name'].duplicated(keep=False), 'name'].unique()
hits = lk[island & lk['name'].isin(dup_names)]

print("=== Internal distances of duplicate names (to determine if they are duplicated) ===")
for nm in hits['name'].unique():
    idx = lk.index[lk['name'] == nm].tolist()
    if len(idx) >= 2:
        d = np.hypot(*(xy[idx[0]] - xy[idx[1]]))
        print(f"{nm}: The two points are {d:.0f} m apart | Addresses: {lk.loc[idx[0],'address']} / {lk.loc[idx[1],'address']}")

print("\n=== Distance overestimation quantification (pipeline value − ideal value) ===")
for i in hits.index:
    d_all = np.hypot(xy[:,0]-xy[i,0], xy[:,1]-xy[i,1])
    d_ideal = np.min(d_all[np.arange(len(lk)) != i])                      # Only exclude self
    mask = (lk['name'] != lk.loc[i,'name']).values                        # Exclude by name (pipeline approach)
    d_pipe = np.min(d_all[mask])
    print(f"{lk.loc[i,'name']} ({lk.loc[i,'adname']}): Ideal {d_ideal:.0f} m, Pipeline {d_pipe:.0f} m, Overestimation {d_pipe-d_ideal:.0f} m")