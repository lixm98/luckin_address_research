import osmnx as ox

# New API: bbox = (west, south, east, north)
bbox = (118.06, 24.43, 118.20, 24.56)

tags = {"aeroway": True}
gdf = ox.features_from_bbox(bbox, tags=tags)

# Only polygon types are retained (runways and aprons are polygon)
gdf = gdf[gdf.geometry.type.isin(['Polygon', 'MultiPolygon'])]

gdf.to_file("xiamen_airport.geojson", driver="GeoJSON")
print(f"get {len(gdf)} airport elements")