import urllib.request

url = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_N24E117_Map.tif"
urllib.request.urlretrieve(url, "xiamen_worldcover_2021.tif")
print("✅ Download completed")