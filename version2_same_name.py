import pandas as pd
for f in ["xiamen_luckin_poi_wgs84.csv", "xiamen_starbucks_poi_wgs84.csv"]:
    df = pd.read_csv(f)
    dup = df['name'].duplicated(keep=False)
    print(f, "| Total rows", len(df), "| Duplicate rows", dup.sum(), "| Unique names involved", df.loc[dup, 'name'].nunique())