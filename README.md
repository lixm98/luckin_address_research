# The Location Pattern of Luckin Coffee on Xiamen Island

Code and data for my MSc Business Analytics dissertation,
Trinity College Dublin, July 2026.

A case-control spatial analysis of Luckin Coffee store locations
on Xiamen Island (176 stores, 880 sampled control points, 60
Starbucks stores as benchmark). The pipeline covers data
collection, control sampling, logistic regression, Moran's I /
LISA diagnostics, GWLR, XGBoost with SHAP, and a pooled
brand-comparison model.

## Repository structure

- `py/collect/`  data collection from the Amap (Gaode) API,
  coordinate transformation (GCJ-02 to WGS-84), metro POI
  splitting, and study-area construction
- `py/v2/`  the final analysis pipeline used in the thesis
- `py/v1/`  earlier iterations, kept for record only
- `csv/`  raw and processed data plus all model outputs
- `geojson/`  district boundaries and sampling-area polygons
- `png/v2/`, `png/v1/`  figures from the v2 and v1 scripts
- `xiamen_worldcover_2021.tif`  ESA WorldCover land-cover raster

Note: `version2_new_MoransI_LISA.py` is the later version of the
LISA script; `version2_MoransI_LISA.py` is an earlier variant.
`py/v1/case_buffer.py` produced `geojson/sampling_area_final.geojson`,
which the v2 pipeline also reads.

## How to reproduce

Python 3.x. Install with `pip install -r requirements.txt`
(a conda `environment.yml` is also provided). Run every script
from the repository root, e.g. `python py/v2/version2_GWLR.py`.
Collection scripts need your own Amap API key. Control sampling
uses a 50 m exclusion ring, 5 controls per case, random seed 42.

## Data note

POI data were retrieved from the Amap (Gaode) Web API on
21 June 2026. Raw responses (`csv/xiamen_*_poi.csv` and other
`csv/xiamen_*` files) and their WGS-84 versions are included, so
the analysis can be reproduced without re-collecting. To
re-collect from scratch, the scripts in `py/collect/` need your
own Amap API key.

## Author

Xingming Li, MSc Business Analytics, Trinity College Dublin.
Supervisor: Professor Chenfeng Pan.
