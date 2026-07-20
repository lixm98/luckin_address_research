import pandas as pd
print("exits:", len(pd.read_csv("xiamen_metro_exits.csv")),
      "| stations:", len(pd.read_csv("xiamen_metro_stations.csv")))