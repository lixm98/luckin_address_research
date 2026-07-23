import requests
import pandas as pd
import time

AMAP_KEY = "e959958de5f40b3052689849233760f8"

KEYWORD = "瑞幸咖啡"
CITY = "厦门"

BASE_URL = "https://restapi.amap.com/v3/place/text"


def fetch_poi_by_keyword(keyword, city, max_pages=50):
    """
    Use Amap keyword search API to capture POI data
    """
    all_pois = []

    for page in range(1, max_pages + 1):
        params = {
            "key": AMAP_KEY,
            "keywords": keyword,
            "city": city,
            "offset": 20,
            "page": page,
            "extensions": "all",
            "output": "json"
        }

        response = requests.get(BASE_URL, params=params, timeout=10)
        data = response.json()

        # When the Amap API request is successful, the status is usually "1"
        if data.get("status") != "1":
            print("Request failed:", data)
            break

        pois = data.get("pois", [])

        if not pois:
            print(f"No. {page} Page has no data, stop.")
            break

        print(f"No. {page} page obtained {len(pois)} piece of data")

        for poi in pois:
            location = poi.get("location", "")
            if location and "," in location:
                lon, lat = location.split(",")
            else:
                lon, lat = None, None

            all_pois.append({
                "name": poi.get("name"),
                "type": poi.get("type"),
                "typecode": poi.get("typecode"),
                "address": poi.get("address"),
                "cityname": poi.get("cityname"),
                "adname": poi.get("adname"),
                "longitude": lon,
                "latitude": lat,
                "location": location,
                "tel": poi.get("tel"),
                "pname": poi.get("pname"),
                "id": poi.get("id")
            })

        time.sleep(0.3)

    return pd.DataFrame(all_pois)


if __name__ == "__main__":
    df = fetch_poi_by_keyword(KEYWORD, CITY)

    print("Original data quantity:", len(df))

    # Simple deduplication: deduplication by name + address + longitude and latitude
    df = df.drop_duplicates(subset=["name", "address", "location"])

    print("Amount of data after deduplication:", len(df))

    df.to_csv("csv/xiamen_luckin_poi.csv", index=False, encoding="utf-8-sig")

    print("saved as csv/xiamen_luckin_poi.csv")