#  Causal GeoXAI: Terracing and Landscape Resilience
Code and Colab notebooks: Causal GeoXAI framework to assess terracing impacts on landscape resilience.
This repository contains the main executable code (Google Earth Engine JavaScript + Python) to reproduce the analysis and outputs presented in:
**Does terracing build landscape resilience? A causal GeoXAI assessment using Earth observations in China’s Yellow River Basin**  
*Submitted to Remote Sensing of Environment*.


## Repository Structure and Corresponding Methods Sections

| Folder     | Language | Methods Section | Description |
|--------|----------|----------------|-------------|
| 'gee_scripts/' | JavaScript (GEE) | 2.2, 2.4 | Data retrieval, preprocessing, LEREI indicator calculation, terrace‑aware RUSLE, export to PyAPI (Colab-Drive/Asset) |
| 'python_scripts/part1_lerei_construction/' | Python | 2.3.1–2.3.2 | Normalisation, Bayesian SEM for LEREI‑X |
| 'python_scripts/part2_causal_inference/' | Python | 2.5–2.6 | DML (ATE), Causal Forest DML (CATE), counterfactual mapping |
| 'python_scripts/part3_geoxai_shap/' | Python | 2.7 | LightGBM, conditioned SHAP, SHAP‑CATE integration |
| 'python_scripts/part4_spatial_analysis/' | Python | 2.3.3, 2.6 | Local Moran’s I, spatial block bootstrap |

## How to Reproduce the Entire Workflow

### Step 1: Google Earth Engine (JavaScript) – Data Preparation

1. Create a [GEE account](https://earthengine.google.com/) and access the [Code Editor](https://code.earthengine.google.com/).
2. Copy the scripts from 'gee_scripts/' into the Code Editor.
3. Run them **in order**:
   - '01_load_and_preprocess.js' – loads and harmonises raw EO data.
   - '02_compute_lerei_indicators.js' – calculates NDVI, NPP, dPC, SOC, climate indices, etc.
   - '03_compute_rusle_terrace.js' – computes baseline and terrace‑adjusted erosion using modified RUSLE.
   - '04_export_training_data.js' – exports the final harmonised rasters to Google Drive or an Asset.
4. Download the exported rasters to your local machine or Google Drive (for Python).

### Step 2: Python – LEREI‑X Construction and Causal Analysis

1. **Set up the Python environment** (local or [Colab](https://colab.research.google.com/)):
   pip install -r requirements.txt

or using conda:
conda env create -f environment.yml
conda activate causal-geoxai

2. Run Python scripts **in order** (all scripts expect the GEE‑exported rasters in data/):
cd python_scripts
- python part1_lerei_construction/01_normalize_indicators.py
- python part1_lerei_construction/02_aggregate_components.py
- python part1_lerei_construction/03_bayesian_sem_lerei_x.py
- python part2_causal_inference/01_dml_ate.py
- python part2_causal_inference/02_causal_forest_cate.py
- python part2_causal_inference/03_counterfactual_map.py
- python part3_geoxai_shap/01_train_lightgbm.py
- python part3_geoxai_shap/02_conditioned_shap.py
- python part3_geoxai_shap/03_shap_cate_integration.py
- python part4_spatial_analysis/morans_i_clustering.py
- python part4_spatial_analysis/spatial_block_bootstrap.py




## Citation
If you use this code, please cite the associated paper (once published) and this repository.

## License
MIT (see LICENSE file)
