import pandas as pd
lk = pd.read_csv("csv/xiamen_luckin_poi_wgs84.csv")
sb = pd.read_csv("csv/xiamen_starbucks_poi_wgs84.csv")
for df, brand in [(lk, "Luckin"), (sb, "Starbucks")]:
    dup_names = df.loc[df['name'].duplicated(keep=False), 'name'].unique()
    island = df[df['adname'].isin(['思明区', '湖里区'])]
    hit = island[island['name'].isin(dup_names)]
    print(brand, "| The name in the case on the island has the same name as that of another store in Chizhong:", len(hit))
    if len(hit): print(hit[['name','adname']].to_string())