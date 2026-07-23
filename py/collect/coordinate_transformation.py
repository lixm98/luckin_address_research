"""
GCJ-02 → WGS-84 coordinate conversion
Add two new columns lon_wgs84 /lat_wgs84 to the original files for all 7 CSVs
"""
import pandas as pd
import math

# GCJ-02 → WGS-84 conversion function
PI = math.pi
A = 6378245.0
EE = 0.00669342162296594323

def _out_of_china(lon, lat):
    return not (73.66 < lon < 135.05 and 3.86 < lat < 53.55)

def _transform_lat(x, y):
    ret = -100.0 + 2.0*x + 3.0*y + 0.2*y*y + 0.1*x*y + 0.2*math.sqrt(abs(x))
    ret += (20.0*math.sin(6.0*x*PI) + 20.0*math.sin(2.0*x*PI)) * 2.0/3.0
    ret += (20.0*math.sin(y*PI) + 40.0*math.sin(y/3.0*PI)) * 2.0/3.0
    ret += (160.0*math.sin(y/12.0*PI) + 320*math.sin(y*PI/30.0)) * 2.0/3.0
    return ret

def _transform_lon(x, y):
    ret = 300.0 + x + 2.0*y + 0.1*x*x + 0.1*x*y + 0.1*math.sqrt(abs(x))
    ret += (20.0*math.sin(6.0*x*PI) + 20.0*math.sin(2.0*x*PI)) * 2.0/3.0
    ret += (20.0*math.sin(x*PI) + 40.0*math.sin(x/3.0*PI)) * 2.0/3.0
    ret += (150.0*math.sin(x/12.0*PI) + 300.0*math.sin(x/30.0*PI)) * 2.0/3.0
    return ret

def gcj02_to_wgs84(lon, lat):
    if _out_of_china(lon, lat):
        return lon, lat
    dlat = _transform_lat(lon - 105.0, lat - 35.0)
    dlon = _transform_lon(lon - 105.0, lat - 35.0)
    radlat = lat / 180.0 * PI
    magic = math.sin(radlat)
    magic = 1 - EE * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((A * (1 - EE)) / (magic * sqrtmagic) * PI)
    dlon = (dlon * 180.0) / (A / sqrtmagic * math.cos(radlat) * PI)
    return lon * 2 - (lon + dlon), lat * 2 - (lat + dlat)


# Process a single file
def convert_file(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    df['longitude'] = df['longitude'].astype(float)
    df['latitude']  = df['latitude'].astype(float)

    wgs = df.apply(lambda r: gcj02_to_wgs84(r['longitude'], r['latitude']), axis=1)
    df['lon_wgs84'] = wgs.map(lambda t: t[0])
    df['lat_wgs84'] = wgs.map(lambda t: t[1])

    df.to_csv(output_csv, index=False, encoding='utf-8-sig')

    # Take a line and compare the offset (use haversine to estimate the number of meters)
    r = df.iloc[0]
    dlon_m = (r['lon_wgs84'] - r['longitude']) * 111000 * math.cos(r['latitude']*PI/180)
    dlat_m = (r['lat_wgs84'] - r['latitude']) * 111000
    shift_m = math.sqrt(dlon_m**2 + dlat_m**2)
    print(f"  {input_csv}: {len(df)} rows  →  {output_csv}  (example offset ≈ {shift_m:.0f}m)")


# Batch processing
FILES = [
    "csv/xiamen_luckin_poi.csv",
    "csv/xiamen_starbucks_poi.csv",
    "csv/xiamen_office.csv",
    "csv/xiamen_residential.csv",
    "csv/xiamen_mall.csv",
    "csv/xiamen_metro_stations.csv",
    "csv/xiamen_metro_exits.csv",
]

for f in FILES:
    output = f.replace(".csv", "_wgs84.csv")
    convert_file(f, output)

print("\n✅ All done. All new files end with _wgs84.csv.")