import pandas as pd

# 1. Check whether there are any Ruixings with the same name in the city
luckin = pd.read_csv("xiamen_luckin_poi_wgs84.csv")
duplicates = luckin['name'].value_counts()
duplicates = duplicates[duplicates > 1]
print(f"Duplicate Ruixing store names: {len(duplicates)}")
print(duplicates.head(10))

# 2. Check the dist_to_nearest_luckin distribution of case in features
features = pd.read_csv("features.csv")
case_dist = features[features['case']==1]['dist_to_nearest_luckin']
print(f"\n=== case 的 dist_to_nearest_luckin ===")
print(f"Minimum: {case_dist.min():.1f}m")
print(f"Median: {case_dist.median():.1f}m")
print(f"Has 0 (It shows that I have not excluded myself): {(case_dist == 0).sum()} 个")
print(f"Has nan (It shows that there are no nearby Luckin stores within 5km): {case_dist.isna().sum()} 个")