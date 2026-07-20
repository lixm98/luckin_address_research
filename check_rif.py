import rasterio
with rasterio.open("xiamen_worldcover_2021.tif") as src:
    print("size:", src.shape)
    print("coordinate system:", src.crs)
    print("boundary:", src.bounds)