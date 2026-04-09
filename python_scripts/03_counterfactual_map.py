# ============================================================================
# Counterfactual Attribution of Terracing Effects (Section 2.6): Year‑specific counterfactual, spatial block bootstrap, positivity
# ============================================================================
# Data‑path neutral: set DATA_DIR and OUTPUT_DIR below.
# Required inputs per year:
#   - LEREI_X_{year}.tif (observed resilience)
#   - CATE_full_{year}.tif (heterogeneous treatment effect)
#   - YRB_terrace_erosion_reduction_{year}_final.tif (treatment intensity)
#   - dml_data_{year}.npz (for positivity diagnostics)
#   - reference_metadata.pkl & valid_mask.npy (from preprocessing)
# ============================================================================

import os
import numpy as np
import pandas as pd
import rasterio
import joblib
from scipy import stats
from rasterio.warp import reproject
import warnings
warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# USER CONFIGURATION – EDIT THIS SECTION
# ----------------------------------------------------------------------------
DATA_DIR = "./data"                     # Directory containing input files
OUTPUT_DIR = "./results"                # Where outputs will be saved
LEREI_DIR = "./lerei_outputs"           # Directory with LEREI_X_{year}.tif rasters
TREAT_DIR = "./gee_exports"             # Directory with terrace erosion reduction rasters
YEARS = [2000, 2010, 2020]              # Benchmark years
BOOTSTRAP_ITER = 1000                   # Number of bootstrap resamples
BLOCK_SIZE = 64                         # Spatial block size for bootstrap
RANDOM_STATE = 42                       # Reproducibility

np.random.seed(RANDOM_STATE)

# ----------------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("COUNTERFACTUAL ATTRIBUTION OF TERRACING EFFECTS")
print("Section 2.6 | Eq. 13–15")
print("=" * 80)

# ----------------------------------------------------------------------------
# Load reference grid and valid mask
# ----------------------------------------------------------------------------
ref_meta_path = os.path.join(DATA_DIR, "reference_metadata.pkl")
valid_mask_path = os.path.join(DATA_DIR, "valid_mask.npy")

if not os.path.exists(ref_meta_path) or not os.path.exists(valid_mask_path):
    raise FileNotFoundError("Missing reference_metadata.pkl or valid_mask.npy in DATA_DIR.")

ref_meta = joblib.load(ref_meta_path)
valid_mask = np.load(valid_mask_path)
shape = ref_meta["shape"]
rows, cols = shape

# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------
def align_to_reference(raster_path):
    """Load raster and align to reference grid (reproject if needed)."""
    with rasterio.open(raster_path) as src:
        arr = src.read(1)
        if arr.shape != shape:
            aligned = np.full(shape, np.nan, dtype=np.float32)
            reproject(
                rasterio.band(src, 1),
                aligned,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref_meta["transform"],
                dst_crs=ref_meta["crs"]
            )
            arr = aligned
    return arr.astype(np.float32)

def save_raster(flat_data, out_name):
    """Save 1D valid-pixel data to GeoTIFF using reference metadata."""
    full = np.full(shape, np.nan, dtype=np.float32)
    full[valid_mask] = flat_data
    out_path = os.path.join(OUTPUT_DIR, f"{out_name}.tif")
    with rasterio.open(
        out_path, "w",
        driver="GTiff",
        height=shape[0],
        width=shape[1],
        count=1,
        dtype="float32",
        crs=ref_meta["crs"],
        transform=ref_meta["transform"],
        nodata=np.nan,
        compress="lzw"
    ) as dst:
        dst.write(full, 1)
    print(f"Saved: {out_path}")

# ----------------------------------------------------------------------------
# 1. Year‑specific counterfactual (Eq. 13–14)
# ----------------------------------------------------------------------------
actual_stack = []
cf_stack = []
gain_stack = []

for year in YEARS:
    print(f"\nProcessing {year}...")

    # Observed LEREI‑X
    y_obs_path = os.path.join(LEREI_DIR, f"LEREI_X_{year}.tif")
    if not os.path.exists(y_obs_path):
        raise FileNotFoundError(f"Missing LEREI raster: {y_obs_path}")
    y_obs = align_to_reference(y_obs_path)[valid_mask]

    # Pixel‑specific CATE τ(X)
    tau_path = os.path.join(OUTPUT_DIR, f"CATE_full_{year}.tif")
    if not os.path.exists(tau_path):
        raise FileNotFoundError(f"Missing CATE raster: {tau_path}")
    tau = align_to_reference(tau_path)[valid_mask]

    # Realized treatment intensity T
    treat_path = os.path.join(TREAT_DIR, f"YRB_terrace_erosion_reduction_{year}_final.tif")
    if not os.path.exists(treat_path):
        raise FileNotFoundError(f"Missing treatment raster: {treat_path}")
    treat = align_to_reference(treat_path)[valid_mask]

    # Eq. 13: Counterfactual LEREI‑X
    y_cf = np.clip(y_obs - tau * treat, 0, 1)

    # Eq. 14: Attributable gain
    gain = y_obs - y_cf

    actual_stack.append(y_obs)
    cf_stack.append(y_cf)
    gain_stack.append(gain)

# Temporal mean across benchmark years
lerei_actual = np.mean(np.stack(actual_stack), axis=0)
lerei_counterfactual = np.mean(np.stack(cf_stack), axis=0)
gain = np.mean(np.stack(gain_stack), axis=0)

# ----------------------------------------------------------------------------
# 2. Spatial block bootstrap for basin‑scale uncertainty
# ----------------------------------------------------------------------------
print("\nRunning spatial block bootstrap...")

# Create spatial block IDs using valid pixel coordinates
valid_indices = np.where(valid_mask.reshape(shape))
coords = np.column_stack(valid_indices)
block_ids = (coords[:, 0] // BLOCK_SIZE, coords[:, 1] // BLOCK_SIZE)
block_df = pd.DataFrame({
    "r": coords[:, 0],
    "c": coords[:, 1],
    "block_r": block_ids[0],
    "block_c": block_ids[1]
})
block_df["block_id"] = block_df["block_r"].astype(str) + "_" + block_df["block_c"].astype(str)

unique_blocks = block_df["block_id"].unique()
gain_boot = []

for b in range(BOOTSTRAP_ITER):
    sampled_blocks = np.random.choice(unique_blocks, size=len(unique_blocks), replace=True)
    idx = block_df[block_df["block_id"].isin(sampled_blocks)].index.values
    gain_boot.append(np.mean(gain[idx]))
    if b % 200 == 0:
        print(f"  Bootstrap {b}/{BOOTSTRAP_ITER}")

gain_boot = np.array(gain_boot)
ci_low = np.percentile(gain_boot, 2.5)
ci_high = np.percentile(gain_boot, 97.5)

# Pixel‑level significance (inter‑annual agreement)
gain_lower = np.percentile(np.stack(gain_stack), 2.5, axis=0)
gain_upper = np.percentile(np.stack(gain_stack), 97.5, axis=0)
significant_mask = gain_lower > 0

# ----------------------------------------------------------------------------
# 3. Positivity / overlap diagnostics (Eq. 15)
# ----------------------------------------------------------------------------
print("\nEvaluating positivity assumption...")
overlap_rows = []

for year in YEARS:
    dml_path = os.path.join(DATA_DIR, f"dml_data_{year}.npz")
    model_path = os.path.join(OUTPUT_DIR, f"LinearDML_{year}.pkl")

    if not os.path.exists(dml_path) or not os.path.exists(model_path):
        print(f"  Warning: skipping positivity for {year} (missing DML data/model)")
        continue

    dml_data = np.load(dml_path, allow_pickle=True)
    X = dml_data["X"].astype(np.float32)
    T = dml_data["T"].astype(np.float32)
    ate_model = joblib.load(model_path)

    # Residualized treatment (Eq. 15)
    t_hat = ate_model.models_t[0][0].predict(X)
    T_res = T - t_hat

    overlap_rows.append({
        "year": year,
        "residual_min": float(T_res.min()),
        "residual_max": float(T_res.max()),
        "residual_sd": float(T_res.std()),
        "p2.5": float(np.percentile(T_res, 2.5)),
        "p97.5": float(np.percentile(T_res, 97.5))
    })

if overlap_rows:
    pd.DataFrame(overlap_rows).to_csv(
        os.path.join(OUTPUT_DIR, "positivity_overlap_diagnostics.csv"), index=False)

# ----------------------------------------------------------------------------
# 4. Summary and export
# ----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"Observed mean LEREI‑X:      {lerei_actual.mean():.5f}")
print(f"Counterfactual mean:        {lerei_counterfactual.mean():.5f}")
print(f"Attributable gain:          {gain.mean():.5f}")
print(f"95% basin‑scale CI:         [{ci_low:.5f}, {ci_high:.5f}]")

pct_sig = significant_mask.mean() * 100
print(f"Pixels with significant gain: {pct_sig:.1f}%")

t_stat, p_val = stats.ttest_rel(lerei_actual, lerei_counterfactual)
print(f"Paired t‑test p‑value:       {p_val:.4e}")

# Save rasters
save_raster(lerei_actual, "LEREI_actual_mean_2000_2020")
save_raster(lerei_counterfactual, "LEREI_counterfactual_mean_2000_2020")
save_raster(gain, "LEREI_gain_mean_2000_2020")
save_raster(significant_mask.astype(np.float32), "LEREI_gain_significant")

# Save summary table
summary_df = pd.DataFrame({
    "metric": [
        "actual_mean",
        "counterfactual_mean",
        "gain_mean",
        "gain_ci_low",
        "gain_ci_high",
        "significant_pixels_percent",
        "paired_t_pvalue"
    ],
    "value": [
        lerei_actual.mean(),
        lerei_counterfactual.mean(),
        gain.mean(),
        ci_low,
        ci_high,
        pct_sig,
        p_val
    ]
})
summary_df.to_csv(os.path.join(OUTPUT_DIR, "counterfactual_summary_section_2_6.csv"), index=False)

print("\n✅ Section 2.6 workflow complete.")
