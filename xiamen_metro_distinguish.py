"""
-stations: typecode == '150500' for count feature
-exits: typecode == '150501', for distance features
"""
import pandas as pd

df = pd.read_csv("xiamen_metro.csv")
df['typecode'] = df['typecode'].astype(str)

# Split
stations = df[df['typecode'] == '150500'].copy()
exits    = df[df['typecode'] == '150501'].copy()

# Remove duplicates by id (if there are duplicates across grids when capturing data before)
stations = stations.drop_duplicates(subset=['id'])
exits    = exits.drop_duplicates(subset=['id'])

# examine
print(f"subway station (150500): {len(stations)} sites")
print(f"Subway entrance and exit (150501): {len(exits)} entrances and exits")
print(f"\n Site sample:")
print(stations[['name', 'adname']].head(5).to_string())
print(f"\n Example of entrance and exit:")
print(exits[['name', 'adname']].head(5).to_string())

# keep
stations.to_csv("xiamen_metro_stations.csv", index=False, encoding='utf-8-sig')
exits.to_csv("xiamen_metro_exits.csv",       index=False, encoding='utf-8-sig')

print("\n✅ Files saved:")
print("   xiamen_metro_stations.csv  →  45 sites for count features")
print("   xiamen_metro_exits.csv     →  ~196 entrances and exits, used for distance features")