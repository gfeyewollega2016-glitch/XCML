# ============================================================================
# Spatial Causal Inference: ATE & CATE for Terrace Erosion Reduction → LEREI‑X
# Manuscript Section 2.5 – Double/Debiased Machine Learning (DML)
# ============================================================================
# Data‑path neutral: set DATA_DIR below to your working directory.
# Required input files per year: dml_data_{year}.npz (see preprocessing script)
# Optional panel file: causal_data_timevarying_panel.csv (for coordinates)
# ============================================================================


import os
import gc
import numpy as np
import pandas as pd
import joblib
import rasterio
from rasterio.transform import from_origin
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from econml.dml import LinearDML, CausalForestDML

# ----------------------------------------------------------------------------
# USER CONFIGURATION – EDIT THIS SECTION
# ----------------------------------------------------------------------------
DATA_DIR = "./data"                     # Directory containing input .npz files
OUTPUT_DIR = "./results"                # Where outputs will be saved
YEARS = [2000, 2010, 2020]              # Benchmark years
RANDOM_STATE = 42                       # For reproducibility

# DML hyperparameters
CV_FOLDS = 5                            # Cross‑fitting folds (ATE)
CV_FOLDS_CF = 3                         # Cross‑fitting folds (CATE)
DELTA_T = 10.0                          # Scale ATE to +10 units
CATE_SAMPLE_SIZE = 30000                # Subsample for Causal Forest training
N_ESTIMATORS_CF = 150                   # Trees in Causal Forest
MIN_SAMPLES_LEAF = 30                   # Honesty constraint
PRED_CHUNK = 100000                     # Chunk size for full‑raster prediction

# ----------------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------
def make_nuisance():
    """RidgeCV nuisance model for m(X) and e(X) (Eq. 9)."""
    return RidgeCV(alphas=np.logspace(-3, 3, 9))

def load_year_npz(year):
    """Load DML‑ready arrays from .npz file."""
    npz_path = os.path.join(DATA_DIR, f"dml_data_{year}.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Missing file: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    Y = data["Y"].astype(np.float32)
    T = data["T"].astype(np.float32)
    X = data["X"].astype(np.float32)
    feature_names = data["feature_names"].tolist()

    # Optional coordinate arrays (if stored in .npz)
    x = data["x"].astype(np.float32) if "x" in data.files else None
    y = data["y"].astype(np.float32) if "y" in data.files else None
    pixel_id = data["pixel_id"].astype(str) if "pixel_id" in data.files else None

    return Y, T, X, feature_names, x, y, pixel_id

def standardize_covariates(X_train):
    """Standardize pre‑treatment covariates (as described in Section 2.5)."""
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X_train).astype(np.float32)
    return Xz, scaler

def get_coordinates(year, n_obs, x_arr=None, y_arr=None, panel_df=None):
    """
    Retrieve pixel coordinates for a given year.
    Priority: coordinates stored in .npz > panel CSV.
    """
    # Case 1: coordinates already in .npz
    if x_arr is not None and y_arr is not None:
        if len(x_arr) != n_obs or len(y_arr) != n_obs:
            raise ValueError(f"Coordinate length mismatch for {year}.")
        df = pd.DataFrame({"x": x_arr, "y": y_arr})
        df["pixel_id"] = df["x"].round(6).astype(str) + "_" + df["y"].round(6).astype(str)
        return df.reset_index(drop=True)

    # Case 2: fallback to panel CSV
    if panel_df is None:
        panel_path = os.path.join(DATA_DIR, "causal_data_timevarying_panel.csv")
        if os.path.exists(panel_path):
            panel_df = pd.read_csv(panel_path)
        else:
            raise FileNotFoundError("No coordinates available. Provide panel CSV or store x/y in .npz.")

    df_year = panel_df[panel_df["year"] == year].copy().reset_index(drop=True)
    if len(df_year) != n_obs:
        raise ValueError(
            f"Panel length mismatch for {year}: panel={len(df_year)}, npz={n_obs}. "
            "Ensure panel order matches npz order, or store x/y in the npz."
        )
    return df_year[["x", "y", "pixel_id"]].reset_index(drop=True)

def infer_transform_from_xy(x_vals, y_vals):
    """Reconstruct affine transform from regular grid of coordinates."""
    xs = np.sort(np.unique(np.round(x_vals.astype(np.float64), 6)))
    ys = np.sort(np.unique(np.round(y_vals.astype(np.float64), 6)))
    if len(xs) < 2 or len(ys) < 2:
        raise ValueError("Need at least 2 unique x and y coordinates.")
    xres = float(np.min(np.diff(xs)))
    yres = float(np.min(np.diff(ys)))
    transform = from_origin(xs.min() - xres / 2.0, ys.max() + yres / 2.0, xres, yres)
    return transform, xs, ys

def save_cate_tiff_from_points(df_map, value_col, out_tif):
    """Export point data to GeoTIFF using inferred grid."""
    df_map = df_map.copy()
    df_map["x_r"] = df_map["x"].round(6)
    df_map["y_r"] = df_map["y"].round(6)

    transform, xs, ys = infer_transform_from_xy(df_map["x_r"].values, df_map["y_r"].values)
    width, height = len(xs), len(ys)
    grid = np.full((height, width), np.nan, dtype=np.float32)

    x_to_col = {x: i for i, x in enumerate(xs)}
    y_to_row = {y: i for i, y in enumerate(ys[::-1])}

    for x, y, v in zip(df_map["x_r"], df_map["y_r"], df_map[value_col]):
        col = x_to_col.get(x)
        row = y_to_row.get(y)
        if row is not None and col is not None:
            grid[row, col] = np.float32(v)

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": transform,
        "nodata": np.nan,
        "compress": "lzw",
    }
    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(grid, 1)

def predict_cate_in_chunks(model, X, chunk_size=PRED_CHUNK):
    """Memory‑efficient CATE prediction for full raster export."""
    preds = []
    for start in range(0, len(X), chunk_size):
        end = min(start + chunk_size, len(X))
        preds.append(model.effect(X[start:end]).astype(np.float32))
        gc.collect()
    return np.concatenate(preds)

# ----------------------------------------------------------------------------
# Load panel (if exists) for coordinate fallback
# ----------------------------------------------------------------------------
panel_path = os.path.join(DATA_DIR, "causal_data_timevarying_panel.csv")
df_panel = pd.read_csv(panel_path) if os.path.exists(panel_path) else None
if df_panel is not None and "pixel_id" not in df_panel.columns:
    df_panel["pixel_id"] = df_panel["x"].astype(str) + "_" + df_panel["y"].astype(str)

# ----------------------------------------------------------------------------
# Main loop: year‑specific DML estimation
# ----------------------------------------------------------------------------
ate_records = []
cate_sample_list = []
feature_importance_list = []

for year in YEARS:
    print(f"\n{'='*60}\nYear {year} – DML Estimation\n{'='*60}")

    Y, T, X, feature_names, x_arr, y_arr, _ = load_year_npz(year)
    coord_df = get_coordinates(year, len(Y), x_arr, y_arr, df_panel)

    print(f"  N = {len(Y):,}  |  Covariates = {X.shape[1]}  |  Treatment mean = {T.mean():.2f} t·ha⁻¹·yr⁻¹")

    # ------------------------------------------------------------------------
    # Standardize covariates (as per manuscript)
    # ------------------------------------------------------------------------
    Xz, scaler = standardize_covariates(X)
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, f"X_scaler_{year}.pkl"))

    # ------------------------------------------------------------------------
    # 2.5.1 ATE: LinearDML (Eq. 8–11)
    # ------------------------------------------------------------------------
    ate_model = LinearDML(
        model_y=make_nuisance(),
        model_t=make_nuisance(),
        cv=CV_FOLDS,
        mc_iters=1,
        discrete_treatment=False,
        random_state=RANDOM_STATE,
    )
    ate_model.fit(Y, T, X=Xz)

    ate_1 = float(np.squeeze(ate_model.ate(X=Xz)))
    ci_low_1, ci_high_1 = ate_model.ate_interval(X=Xz, alpha=0.05)
    ci_low_1, ci_high_1 = float(np.squeeze(ci_low_1)), float(np.squeeze(ci_high_1))

    ate_10 = ate_1 * DELTA_T
    ci_low_10 = ci_low_1 * DELTA_T
    ci_high_10 = ci_high_1 * DELTA_T

    ate_records.append({
        "year": year,
        "ATE_per_1_unit": ate_1,
        "ATE_per_10_units": ate_10,
        "CI_lower_per_10": ci_low_10,
        "CI_upper_per_10": ci_high_10,
        "n_obs": len(Y),
        "treatment_mean": float(T.mean()),
        "treatment_min": float(T.min()),
        "treatment_max": float(T.max()),
    })

    print(f"  ✅ ATE per +10 units: {ate_10:.6f}  (95% CI: [{ci_low_10:.6f}, {ci_high_10:.6f}])")
    joblib.dump(ate_model, os.path.join(OUTPUT_DIR, f"LinearDML_{year}.pkl"))

    # ------------------------------------------------------------------------
    # 2.5.2 CATE: CausalForestDML (Eq. 12)
    # ------------------------------------------------------------------------
    np.random.seed(RANDOM_STATE)
    n_sample = min(CATE_SAMPLE_SIZE, len(Y))
    idx_sample = np.random.choice(len(Y), n_sample, replace=False)

    X_s = Xz[idx_sample]
    Y_s = Y[idx_sample]
    T_s = T[idx_sample]
    coord_s = coord_df.iloc[idx_sample].copy().reset_index(drop=True)

    cf_model = CausalForestDML(
        model_y=make_nuisance(),
        model_t=make_nuisance(),
        cv=CV_FOLDS_CF,
        mc_iters=1,
        n_estimators=N_ESTIMATORS_CF,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        max_depth=8,
        discrete_treatment=False,
        inference=False,
        random_state=RANDOM_STATE,
    )
    cf_model.fit(Y_s, T_s, X=X_s)
    joblib.dump(cf_model, os.path.join(OUTPUT_DIR, f"CausalForestDML_{year}.pkl"))

    # Feature importances
    fi_df = pd.DataFrame({
        "year": year,
        "feature": feature_names,
        "importance": cf_model.feature_importances_
    }).sort_values("importance", ascending=False)
    feature_importance_list.append(fi_df)

    # Sampled CATE for diagnostics
    cate_sample = cf_model.effect(X_s).astype(np.float32)
    sample_df = pd.DataFrame({
        "x": coord_s["x"].values,
        "y": coord_s["y"].values,
        "year": year,
        "pixel_id": coord_s["pixel_id"].values,
        "cate": cate_sample,
    })
    sample_df["cate_quartile"] = pd.qcut(sample_df["cate"], 4,
                                         labels=["Q1_low", "Q2", "Q3", "Q4_high"])
    cate_sample_list.append(sample_df)

    # ------------------------------------------------------------------------
    # Full‑year CATE map & GeoTIFF export
    # ------------------------------------------------------------------------
    cate_full = predict_cate_in_chunks(cf_model, Xz)

    df_year_full = coord_df[["x", "y", "pixel_id"]].copy()
    if len(df_year_full) == len(cate_full):
        df_year_full["cate"] = cate_full
        df_year_full["cate_quartile"] = pd.qcut(df_year_full["cate"], 4,
                                                labels=["Q1_low", "Q2", "Q3", "Q4_high"])
        df_year_full.to_csv(os.path.join(OUTPUT_DIR, f"CATE_full_{year}.csv"), index=False)
        tif_path = os.path.join(OUTPUT_DIR, f"CATE_full_{year}.tif")
        save_cate_tiff_from_points(df_year_full, "cate", tif_path)
        print(f"  ✅ Full CATE map exported: {tif_path}")
    else:
        print(f"  ⚠️ Length mismatch; skipping GeoTIFF export.")

    gc.collect()

# ----------------------------------------------------------------------------
# Save summary tables
# ----------------------------------------------------------------------------
if ate_records:
    pd.DataFrame(ate_records).to_csv(os.path.join(OUTPUT_DIR, "ATE_yearly.csv"), index=False)
if cate_sample_list:
    pd.concat(cate_sample_list, ignore_index=True).to_csv(
        os.path.join(OUTPUT_DIR, "CATE_yearly_sampled.csv"), index=False)
if feature_importance_list:
    pd.concat(feature_importance_list, ignore_index=True).to_csv(
        os.path.join(OUTPUT_DIR, "CF_feature_importance_yearly.csv"), index=False)

print("\n" + "="*70)
print("✅ Year‑by‑year DML analysis complete.")
print(f"Outputs saved in: {OUTPUT_DIR}")
print("  - ATE_yearly.csv")
print("  - CATE_yearly_sampled.csv")
print("  - CF_feature_importance_yearly.csv")
print("  - CATE_full_<year>.csv and .tif")
print("="*70)
