"""
LEREI‑X Feature Contribution Analysis with LightGBM + SHAP
===========================================================
(Manuscript Section 2.7): Global and year‑specific SHAP explanations.

Data‑path neutral: set DATA_DIR to the root folder containing all input rasters.
The expected sub‑directory structure is:

DATA_DIR/
├── LEREI-X_Results/
│   ├── LEREI-X_1990.tif
│   ├── LEREI-X_2000.tif
│   ├── LEREI-X_2010.tif
│   ├── LEREI-X_2020.tif
│   └── LEREI-X_2025.tif
├── Data1/
│   ├── 1990_dPC.tif
│   ├── 2000_dPC.tif
│   ├── median_ndvi_1985_1990.tif
│   ├── median_ndvi_1990_2000.tif
│   ├── NPP_1990.tif
│   ├── NTLI_1990.tif
│   └── dNBR_1990.tif
│   └── ...
├── Climate/
│   ├── climate_1990.tif
│   ├── climate_2000.tif
│   ├── SPEI_mean_1985_1990.tif
│   └── ...
├── Soil&Erosion/
│   ├── SOC_1990.tif
│   ├── SOC_2000.tif
│   ├── ErosionRisk_1990.tif
│   └── ...
├── Slope.tif
├── TRI_Riley.tif
└── ...

Adjust the paths inside STATIC_PATHS and TIME_VARYING if your layout differs.
All outputs are saved to OUTPUT_DIR (default: ./results/SHAP).
"""

import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# USER CONFIGURATION – CHANGE THESE PATHS
# =============================================================================
DATA_DIR = "./data"                     # Root directory containing all input rasters
OUTPUT_DIR = "./results/SHAP"           # Where outputs will be saved

# -----------------------------------------------------------------------------
# LEREI‑X outcome rasters (one per year)
# -----------------------------------------------------------------------------
LEREI_PATHS = {
    1990: os.path.join(DATA_DIR, "LEREI-X_Results/LEREI-X_1990.tif"),
    2000: os.path.join(DATA_DIR, "LEREI-X_Results/LEREI-X_2000.tif"),
    2010: os.path.join(DATA_DIR, "LEREI-X_Results/LEREI-X_2010.tif"),
    2020: os.path.join(DATA_DIR, "LEREI-X_Results/LEREI-X_2020.tif"),
    2025: os.path.join(DATA_DIR, "LEREI-X_Results/LEREI-X_2025.tif"),
}

# -----------------------------------------------------------------------------
# Static predictors (same for all years)
# -----------------------------------------------------------------------------
STATIC_PATHS = {
    "Slope": os.path.join(DATA_DIR, "Slope.tif"),
    "TRI":   os.path.join(DATA_DIR, "TRI_Riley.tif"),
}

# -----------------------------------------------------------------------------
# Time‑varying predictors per year
# -----------------------------------------------------------------------------
TIME_VARYING = {
    1990: {
        "dpc":    os.path.join(DATA_DIR, "Data1/1990_dPC.tif"),
        "NDVI":   os.path.join(DATA_DIR, "Data1/median_ndvi_1985_1990.tif"),
        "NPP":    os.path.join(DATA_DIR, "Data1/NPP_1990.tif"),
        "NTLI":   os.path.join(DATA_DIR, "Data1/NTLI_1990.tif"),
        "Fire":   os.path.join(DATA_DIR, "Data1/dNBR_1990.tif"),
        "Climate": os.path.join(DATA_DIR, "Climate/climate_1990.tif"),
        "SPEI":   os.path.join(DATA_DIR, "Climate/SPEI_mean_1985_1990.tif"),
        "SOC":    os.path.join(DATA_DIR, "Soil/SOC_1990.tif"),
        "Erosion": os.path.join(DATA_DIR, "Erosion/ErosionRisk_1990.tif"),
    },
    2000: {
        "dpc":    os.path.join(DATA_DIR, "Data1/2000_dPC.tif"),
        "NDVI":   os.path.join(DATA_DIR, "Data1/median_ndvi_1990_2000.tif"),
        "NPP":    os.path.join(DATA_DIR, "Data1/NPP_2000.tif"),
        "NTLI":   os.path.join(DATA_DIR, "Data1/NTLI_2000.tif"),
        "Fire":   os.path.join(DATA_DIR, "Data1/dNBR_2000.tif"),
        "Climate": os.path.join(DATA_DIR, "Climate/climate_2000.tif"),
        "SPEI":   os.path.join(DATA_DIR, "Climate/SPEI_mean_1990_2000.tif"),
        "SOC":    os.path.join(DATA_DIR, "Soil/SOC_2000.tif"),
        "Erosion": os.path.join(DATA_DIR, "Erosion/ErosionRisk_2000.tif"),
    },
    2010: {
        "dpc":    os.path.join(DATA_DIR, "Data1/2010_dPC.tif"),
        "NDVI":   os.path.join(DATA_DIR, "Data1/median_ndvi_2000_2010.tif"),
        "NPP":    os.path.join(DATA_DIR, "Data1/NPP_2010.tif"),
        "NTLI":   os.path.join(DATA_DIR, "Data1/NTLI_2010.tif"),
        "Fire":   os.path.join(DATA_DIR, "Data1/dNBR_2010.tif"),
        "Climate": os.path.join(DATA_DIR, "Climate/climate_2010.tif"),
        "SPEI":   os.path.join(DATA_DIR, "Climate/SPEI_mean_2000_2010.tif"),
        "SOC":    os.path.join(DATA_DIR, "Soil/SOC_2010.tif"),
        "Erosion": os.path.join(DATA_DIR, "Erosion/ErosionRisk_2010.tif"),
    },
    2020: {
        "dpc":    os.path.join(DATA_DIR, "Data1/2020_dPC.tif"),
        "NDVI":   os.path.join(DATA_DIR, "Data1/median_ndvi_2010_2020.tif"),
        "NPP":    os.path.join(DATA_DIR, "Data1/NPP_2020.tif"),
        "NTLI":   os.path.join(DATA_DIR, "Data1/NTLI_2020.tif"),
        "Fire":   os.path.join(DATA_DIR, "Data1/dNBR_2020.tif"),
        "Climate": os.path.join(DATA_DIR, "Climate/climate_2020.tif"),
        "SPEI":   os.path.join(DATA_DIR, "Climate/SPEI_mean_2010_2020.tif"),
        "SOC":    os.path.join(DATA_DIR, "Soil/SOC_2020.tif"),
        "Erosion": os.path.join(DATA_DIR, "Erosion/ErosionRisk_2020.tif"),
    },
    2025: {
        "dpc":    os.path.join(DATA_DIR, "Data1/2025_dPC.tif"),
        "NDVI":   os.path.join(DATA_DIR, "Data1/median_ndvi_2020_2025.tif"),
        "NPP":    os.path.join(DATA_DIR, "Data1/NPP_2025.tif"),
        "NTLI":   os.path.join(DATA_DIR, "Data1/NTLI_2025.tif"),
        "Fire":   os.path.join(DATA_DIR, "Data1/dNBR_2025.tif"),
        "Climate": os.path.join(DATA_DIR, "Climate/climate_2025.tif"),
        "SPEI":   os.path.join(DATA_DIR, "Climate/SPEI_mean_2020_2025.tif"),
        "SOC":    os.path.join(DATA_DIR, "Soil/SOC_2025.tif"),#2022
        "Erosion": os.path.join(DATA_DIR, "Erosion/ErosionRisk_2025.tif"),
    },
}

# =============================================================================
# CONSTANTS (adjust if needed)
# =============================================================================
YEARS = [1990, 2000, 2010, 2020, 2025]
RANDOM_STATE = 42
MAX_SAMPLES_PER_YEAR = 50000
MAX_BACKGROUND_SHAP = 1000
MAX_SHAP_SAMPLE = 10000
MAX_GLOBAL_SHAP_SAMPLE = 5000

LGB_PARAMS = {
    "n_estimators": 400,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbose": -1,
}

FEATURE_COLS = [
    "Slope", "TRI", "SOC", "Erosion", "dpc", "NDVI", "NPP",
    "NTLI", "Fire", "MAT", "MAP", "SPEI"
]

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def align_to_reference(src_path, ref_meta, band=1, resampling=Resampling.bilinear):
    """Reproject a raster to match the reference grid."""
    with rasterio.open(src_path) as src:
        dst = np.full((ref_meta["height"], ref_meta["width"]), np.nan, dtype=np.float32)
        reproject(
            source=rasterio.band(src, band),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_meta["transform"],
            dst_crs=ref_meta["crs"],
            resampling=resampling,
            src_nodata=src.nodata,
            dst_nodata=np.nan,
        )
    return dst

def align_multiband_to_reference(src_path, ref_meta, bands=(1, 2), resampling=Resampling.bilinear):
    """ Align multiple bands of a raster to a reference grid."""
    out = []
    for b in bands:
        out.append(align_to_reference(src_path, ref_meta, band=b, resampling=resampling))
    return out

def build_year_dataframe(year, ref_meta, max_samples=30000):
    """Construct a DataFrame for a single year with all predictors and LEREI‑X."""
    # Outcome
    y_arr = align_to_reference(LEREI_PATHS[year], ref_meta, resampling=Resampling.bilinear)

    # Static predictors
    static_arrays = {
        name: align_to_reference(path, ref_meta, resampling=Resampling.bilinear)
        for name, path in STATIC_PATHS.items()
    }

    # Time‑varying predictors
    tpaths = TIME_VARYING[year]
    dpc = align_to_reference(tpaths["dpc"], ref_meta, resampling=Resampling.bilinear)
    ndvi = align_to_reference(tpaths["NDVI"], ref_meta, resampling=Resampling.bilinear)
    npp = align_to_reference(tpaths["NPP"], ref_meta, resampling=Resampling.bilinear)
    ntli = align_to_reference(tpaths["NTLI"], ref_meta, resampling=Resampling.bilinear)
    fire = align_to_reference(tpaths["Fire"], ref_meta, resampling=Resampling.bilinear)
    mat, map_ = align_multiband_to_reference(tpaths["Climate"], ref_meta, bands=(1, 2), resampling=Resampling.bilinear)
    spei = align_to_reference(tpaths["SPEI"], ref_meta, resampling=Resampling.bilinear)
    soc = align_to_reference(tpaths["SOC"], ref_meta, resampling=Resampling.bilinear)
    erosion = align_to_reference(tpaths["Erosion"], ref_meta, resampling=Resampling.bilinear)

    df = pd.DataFrame({
        "LEREI_X": y_arr.ravel(),
        "Slope": static_arrays["Slope"].ravel(),
        "TRI": static_arrays["TRI"].ravel(),
        "SOC": soc.ravel(),
        "Erosion": erosion.ravel(),
        "dpc": dpc.ravel(),
        "NDVI": ndvi.ravel(),
        "NPP": npp.ravel(),
        "NTLI": ntli.ravel(),
        "Fire": fire.ravel(),
        "MAT": mat.ravel(),
        "MAP": map_.ravel(),
        "SPEI": spei.ravel(),
        "year": year
    })

    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=RANDOM_STATE + year)
    return df

def train_lightgbm(X_train, y_train, X_test, y_test):
    """Train LightGBM regressor and return model + test metrics."""
    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    r2 = r2_score(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    return model, r2, rmse

def compute_shap_importance(model, X_background, X_shap):
    """Compute mean absolute SHAP and % contribution."""
    explainer = shap.TreeExplainer(model, data=X_background)
    shap_values = explainer.shap_values(X_shap)
    mean_abs = np.abs(shap_values).mean(axis=0)
    pct = 100 * mean_abs / mean_abs.sum()
    importance = pd.DataFrame({
        "feature": FEATURE_COLS,
        "mean_abs_SHAP": mean_abs,
        "contribution_pct": pct
    }).sort_values("contribution_pct", ascending=False)
    return explainer, shap_values, importance

def plot_global_shap_figure(importance_df, shap_values, X_shap, out_path):
    """Create multi‑panel figure: (a) bar plot with % labels, (b) violin summary."""
    fig = plt.figure(figsize=(15, 6))
    ax1 = plt.subplot(1, 2, 1)
    imp_sorted = importance_df.sort_values("mean_abs_SHAP", ascending=True)
    bars = ax1.barh(imp_sorted["feature"], imp_sorted["mean_abs_SHAP"])
    for bar, pct_val in zip(bars, imp_sorted["contribution_pct"]):
        ax1.text(bar.get_width() + 0.0005, bar.get_y() + bar.get_height()/2,
                 f"{pct_val:.1f}%", va="center", fontsize=11)
    ax1.set_title("(a) Global feature ranking", loc="left", fontsize=13)
    ax1.set_xlabel("Mean |SHAP value|")
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)

    ax2 = plt.subplot(1, 2, 2)
    shap.summary_plot(shap_values, X_shap, feature_names=FEATURE_COLS,
                      show=False, plot_type="violin", max_display=12)
    ax2.set_title("(b) SHAP violin summary", loc="left", fontsize=12)
    plt.tight_layout(w_pad=3)
    plt.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"Saved global SHAP figure: {out_path}")

# =============================================================================
# MAIN
# =============================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Reference grid from first LEREI raster
    with rasterio.open(LEREI_PATHS[1990]) as ref:
        ref_meta = {
            "transform": ref.transform,
            "crs": ref.crs,
            "height": ref.height,
            "width": ref.width,
        }

    print("Building pooled dataset...")
    all_dfs = []
    for yr in YEARS:
        print(f"  Processing {yr}...")
        df_yr = build_year_dataframe(yr, ref_meta, max_samples=MAX_SAMPLES_PER_YEAR)
        all_dfs.append(df_yr)
    df_pooled = pd.concat(all_dfs, ignore_index=True)
    print(f"Pooled sample size: {df_pooled.shape[0]}")

    X = df_pooled[FEATURE_COLS]
    y = df_pooled["LEREI_X"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    print("Training LightGBM...")
    model, r2, rmse = train_lightgbm(X_train, y_train, X_test, y_test)
    print(f"Test R² = {r2:.4f}, RMSE = {rmse:.4f}")

    print("Computing global SHAP...")
    bg = shap.sample(X_train, min(MAX_BACKGROUND_SHAP, len(X_train)), random_state=RANDOM_STATE)
    X_shap_global = shap.sample(X_test, min(MAX_GLOBAL_SHAP_SAMPLE, len(X_test)), random_state=RANDOM_STATE)
    explainer, shap_vals_global, importance_df = compute_shap_importance(model, bg, X_shap_global)

    # Save importance CSV
    importance_df.to_csv(os.path.join(OUTPUT_DIR, "LEREI_X_SHAP_feature_contributions.csv"), index=False)

    # Save per‑sample SHAP values
    shap_sample_df = pd.DataFrame(shap_vals_global, columns=[f"SHAP_{c}" for c in FEATURE_COLS])
    shap_sample_df["predicted_LEREI_X"] = model.predict(X_shap_global)
    shap_sample_df["observed_LEREI_X"] = y_test.loc[X_shap_global.index].values
    shap_sample_df["year"] = df_pooled.loc[X_shap_global.index, "year"].values
    shap_sample_df.to_csv(os.path.join(OUTPUT_DIR, "LEREI_X_SHAP_samples.csv"), index=False)

    # Global multi‑panel figure
    plot_global_shap_figure(importance_df, shap_vals_global, X_shap_global,
                            os.path.join(OUTPUT_DIR, "Fig_Global_SHAP_MultiPanel.png"))

    # Year‑wise SHAP beeswarm plots
    beeswarm_dir = os.path.join(OUTPUT_DIR, "SHAP_Beeswarm")
    os.makedirs(beeswarm_dir, exist_ok=True)

    for yr in YEARS:
        print(f"Generating SHAP beeswarm for {yr}...")
        df_yr = build_year_dataframe(yr, ref_meta, max_samples=MAX_SHAP_SAMPLE)
        X_yr = df_yr[FEATURE_COLS]
        if len(X_yr) == 0:
            print(f"  Warning: no valid samples for {yr}, skipping.")
            continue
        X_plot = shap.sample(X_yr, min(3000, len(X_yr)), random_state=RANDOM_STATE)
        shap_vals_yr = explainer.shap_values(X_plot)

        plt.figure(figsize=(8, 6))
        shap.summary_plot(shap_vals_yr, X_plot, feature_names=FEATURE_COLS,
                          show=False, max_display=12)
        plt.title(f"SHAP Beeswarm – LEREI‑X ({yr})", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(beeswarm_dir, f"SHAP_Beeswarm_{yr}.png"), dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {os.path.join(beeswarm_dir, f'SHAP_Beeswarm_{yr}.png')}")

    print("\n✅ SHAP analysis complete. Outputs saved to:", OUTPUT_DIR)


    if __name__ == "__main__":
          main()
