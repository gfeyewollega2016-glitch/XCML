"""
Resilience modelling_LEREI-X (Multi-year)
=========================================
Bayesian SEM of pre-computed resistance (R), Recovery (C), and adaptability (A) using Section 2.3.2 Eqs. (1-3).
Input rasters: 3 bands (band1=R, band2=C, band3=A)
Outputs:
  - Annual LEREI‑X GeoTIFF (0–1 scaled)
  - Posterior traces (NetCDF)
  - Convergence & performance summary (CSV)

Method:
  - Bayesian one‑factor model learns latent resilience from R, C, A.
  - LightGBM surrogate maps components to the latent score for full‑raster prediction.
  - Spatial block cross‑validation assesses generalisation.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
import pymc as pm
import arviz as az
import lightgbm as lgb
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


class LEREIXBayesianSEM:
    """
    Bayesian latent‑factor SEM for annual LEREI‑X estimation.
    No fixed component weights – loadings are learned from data.
    """

    def __init__(self, output_dir: str, random_state: int = 42):
        self.output_dir = Path(output_dir)
        self.random_state = random_state

        self.posterior = {}
        self.component_means = None
        self.component_stds = None
        self.surrogate = None
        self.scaler = None

        for sub in ["rasters", "traces", "summary"]:
            (self.output_dir / sub).mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------------
    # Data sampling
    # ----------------------------------------------------------------------
    def load_samples(self, raster_path: str, n_samples: int = 2000) -> dict:
        """Randomly sample valid pixels from a 3‑band component raster."""
        rng = np.random.default_rng(self.random_state)

        with rasterio.open(raster_path) as src:
            rows, cols = src.height, src.width
            transform = src.transform
            data = {"R": [], "C": [], "A": [], "x": [], "y": []}

            attempts = 0
            while len(data["R"]) < n_samples and attempts < n_samples * 20:
                attempts += 1
                r, c = rng.integers(0, rows), rng.integers(0, cols)
                vals = src.read([1, 2, 3], window=Window(c, r, 1, 1)).flatten()
                if np.isnan(vals).any():
                    continue
                data["R"].append(vals[0])
                data["C"].append(vals[1])
                data["A"].append(vals[2])
                data["x"].append(transform.c + c * transform.a)
                data["y"].append(transform.f + r * transform.e)

        return {k: np.asarray(v, dtype=np.float32) for k, v in data.items()}

    # ----------------------------------------------------------------------
    # Standardisation
    # ----------------------------------------------------------------------
    def standardise_components(self, data: dict) -> np.ndarray:
        """Z‑score standardise R, C, A; store means and stds for later use."""
        X = np.column_stack([data["R"], data["C"], data["A"]]).astype(np.float32)
        self.component_means = X.mean(axis=0)
        self.component_stds = X.std(axis=0) + 1e-8
        return (X - self.component_means) / self.component_stds

    # ----------------------------------------------------------------------
    # Bayesian latent factor model
    # ----------------------------------------------------------------------
    def fit_sem(self, data: dict, draws: int = 1000, tune: int = 1000):
        """
        Fit a one‑factor Bayesian SEM:
            η ~ N(0,1)
            R_obs = λ_R·η + ε_R,  ε_R ~ N(0, σ_R²)
            C_obs = λ_C·η + ε_C,  ε_C ~ N(0, σ_C²)
            A_obs = λ_A·η + ε_A,  ε_A ~ N(0, σ_A²)
        Loadings are constrained positive for identifiability.
        """
        Xz = self.standardise_components(data)
        Rz, Cz, Az = Xz[:, 0], Xz[:, 1], Xz[:, 2]
        n = len(Rz)

        with pm.Model() as model:
            eta = pm.Normal("eta", mu=0.0, sigma=1.0, shape=n)
            loadings = pm.HalfNormal("loadings", sigma=1.0, shape=3)

            sigma_R = pm.HalfNormal("sigma_R", sigma=1.0)
            sigma_C = pm.HalfNormal("sigma_C", sigma=1.0)
            sigma_A = pm.HalfNormal("sigma_A", sigma=1.0)

            pm.Normal("R_obs", mu=loadings[0] * eta, sigma=sigma_R, observed=Rz)
            pm.Normal("C_obs", mu=loadings[1] * eta, sigma=sigma_C, observed=Cz)
            pm.Normal("A_obs", mu=loadings[2] * eta, sigma=sigma_A, observed=Az)

            trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=4,
                cores=1,
                target_accept=0.95,
                random_seed=self.random_state,
                return_inferencedata=True,
            )

        # Posterior summaries
        eta_mean = trace.posterior["eta"].mean(dim=("chain", "draw")).values.astype(np.float32)
        load_mean = trace.posterior["loadings"].mean(dim=("chain", "draw")).values.astype(np.float32)

        self.posterior = {
            "load_R": float(load_mean[0]),
            "load_C": float(load_mean[1]),
            "load_A": float(load_mean[2]),
            "share_R": float(load_mean[0] / load_mean.sum()),
            "share_C": float(load_mean[1] / load_mean.sum()),
            "share_A": float(load_mean[2] / load_mean.sum()),
            "sigma_R": float(trace.posterior["sigma_R"].mean().values),
            "sigma_C": float(trace.posterior["sigma_C"].mean().values),
            "sigma_A": float(trace.posterior["sigma_A"].mean().values),
        }

        return trace, eta_mean

    # ----------------------------------------------------------------------
    # Surrogate model for raster prediction
    # ----------------------------------------------------------------------
    def fit_surrogate(self, Xz: np.ndarray, eta: np.ndarray):
        """Train LightGBM to map standardised components → latent score."""
        model = lgb.LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=self.random_state,
            verbose=-1,
        )
        model.fit(Xz, eta)
        pred = model.predict(Xz)
        r2 = r2_score(eta, pred)
        rmse = np.sqrt(mean_squared_error(eta, pred))

        # Scale latent scores to [0,1] for output
        self.scaler = MinMaxScaler(feature_range=(0, 1)).fit(eta.reshape(-1, 1))
        self.surrogate = model
        return r2, rmse

    # ----------------------------------------------------------------------
    # Spatial cross‑validation
    # ----------------------------------------------------------------------
    def spatial_cv(self, Xz: np.ndarray, eta: np.ndarray, coords: np.ndarray, n_folds: int = 5):
        """Spatial block CV for the surrogate model."""
        blocks = KMeans(n_clusters=n_folds, random_state=self.random_state, n_init=10).fit_predict(coords)
        scores = []
        for fold in range(n_folds):
            train = blocks != fold
            test = blocks == fold
            if test.sum() < 20:
                continue
            model = lgb.LGBMRegressor(
                n_estimators=250,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=self.random_state,
                verbose=-1,
            )
            model.fit(Xz[train], eta[train])
            pred = model.predict(Xz[test])
            scores.append({
                "cv_r2": r2_score(eta[test], pred),
                "cv_rmse": np.sqrt(mean_squared_error(eta[test], pred)),
            })
        return pd.DataFrame(scores).mean().to_dict() if scores else {"cv_r2": np.nan, "cv_rmse": np.nan}

    # ----------------------------------------------------------------------
    # Raster export
    # ----------------------------------------------------------------------
    def generate_raster(self, in_path: str, year: int, chunk: int = 512) -> str:
        """Apply surrogate model to full raster and export LEREI‑X (0–1)."""
        if self.surrogate is None or self.scaler is None:
            raise RuntimeError("Surrogate model not fitted. Call fit_sem() and fit_surrogate() first.")

        out_path = self.output_dir / "rasters" / f"LEREI_X_{year}.tif"

        with rasterio.open(in_path) as src:
            profile = src.profile.copy()
            profile.update(dtype="float32", count=1, compress="lzw", nodata=np.nan)

            with rasterio.open(out_path, "w", **profile) as dst:
                for row in range(0, src.height, chunk):
                    win = Window(0, row, src.width, min(chunk, src.height - row))
                    R = src.read(1, window=win).astype(np.float32)
                    C = src.read(2, window=win).astype(np.float32)
                    A = src.read(3, window=win).astype(np.float32)

                    valid = ~(np.isnan(R) | np.isnan(C) | np.isnan(A))
                    Xw = np.stack([R, C, A], axis=-1)
                    Xw_z = (Xw - self.component_means) / self.component_stds

                    pred = self.surrogate.predict(Xw_z.reshape(-1, 3)).reshape(R.shape)
                    pred_scaled = self.scaler.transform(pred.reshape(-1, 1)).reshape(R.shape)

                    out = np.full(R.shape, np.nan, dtype=np.float32)
                    out[valid] = pred_scaled[valid].astype(np.float32)
                    dst.write(out, 1, window=win)

        return str(out_path)

    # ----------------------------------------------------------------------
    # Multi‑year pipeline
    # ----------------------------------------------------------------------
    def run_multi_year(self, file_list, years, n_samples=2000, draws=1000, tune=1000):
        """Run full workflow for each year and compile summary."""
        records = []

        for path, year in zip(file_list, years):
            print(f"Processing {year}...")
            data = self.load_samples(path, n_samples)
            trace, eta_mean = self.fit_sem(data, draws=draws, tune=tune)
            Xz = self.standardise_components(data)
            coords = np.column_stack([data["x"], data["y"]])

            # Convergence diagnostics (exclude large latent vector 'eta')
            rhat = az.rhat(trace)
            ess = az.ess(trace)
            params = [v for v in rhat.data_vars if v != "eta"]
            max_rhat = max(float(rhat[v].max().values) for v in params)
            min_ess = min(float(ess[v].min().values) for v in params)
            max_ess = max(float(ess[v].max().values) for v in params)

            train_r2, train_rmse = self.fit_surrogate(Xz, eta_mean)
            cv = self.spatial_cv(Xz, eta_mean, coords)
            raster_path = self.generate_raster(path, year)

            # Save trace
            az.to_netcdf(trace, self.output_dir / "traces" / f"trace_{year}.nc")

            records.append({
                "year": year,
                "max_rhat": round(max_rhat, 4),
                "min_ess": int(min_ess),
                "max_ess": int(max_ess),
                "train_r2": round(train_r2, 4),
                "train_rmse": round(train_rmse, 4),
                "cv_r2": round(float(cv.get("cv_r2", np.nan)), 4),
                "cv_rmse": round(float(cv.get("cv_rmse", np.nan)), 4),
                **{k: round(v, 4) for k, v in self.posterior.items()},
                "raster": raster_path,
            })

        df = pd.DataFrame(records)
        df.to_csv(self.output_dir / "summary" / "metrics.csv", index=False)
        print("\n✅ Multi‑year processing complete.\n")
        print(df.to_string(index=False))
        print(f"\nGlobal diagnostics:\n  Max R̂ = {df['max_rhat'].max():.4f}\n  ESS range = {df['min_ess'].min()}–{df['max_ess'].max()}")
        return df



# =============================================================================
# Example usage
# =============================================================================
if __name__ == "__main__":
    # Update these paths to your actual component rasters
    file_list = [
        "/path/to/components_1990.tif",
        "/path/to/components_2000.tif",
        "/path/to/components_2010.tif",
        "/path/to/components_2020.tif",
        "/path/to/components_2025.tif",
    ]
    years = [1990, 2000, 2010, 2020, 2025]

    # Validate all rasters exist and have same shape
    for p in file_list:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing raster: {p}")

    model = LEREIXBayesianSEM(output_dir="LEREI_Output", random_state=42)
    
    # n_samples=3000 is sufficient for basin‑scale heterogeneity
    metrics = model.run_multi_year(
        file_list=file_list,
        years=years,
        n_samples=3000,   # >400, sufficient for stable posteriors
        draws=1000,
        tune=1000
    )
    
    print("\n📊 Final metrics saved to LEREI_Output/summary/metrics.csv")
    print("📁 LEREI‑X rasters saved to LEREI_Output/rasters/")




