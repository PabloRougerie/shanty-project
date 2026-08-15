"""Fit the universal LGBM g(features, h) and calibrate R90 lookups (LGBM + baseline).

Produces only the artifacts: model pickle, two R90 lookups,
and the calibration predictions behind them.

Steps performed by train():
1. Load df_train + df_val, concatenate into df_trainval, build long-format features.
2. Fit the FINAL LGBM on df_trainval (train+val combined) — this is the model that
   gets pickled and served. More data in, more accurate in production.
3. Calibration (LGBM): refit a SEPARATE LGBM on train only, predict on the held-out
   val rows, and compute R(p, h, s) from those residuals via fit_conditional_radius.
   This model is refit on purpose: the final model from step 2 has already seen val
   during its own fit, so its val predictions would be in-sample and understate the
   true error — the R90 radius would be miscalibrated (too tight).
4. Calibration (baseline): same held-out val rows, but predict_baseline is a
   closed-form constant-velocity extrapolation, so no fit is needed.
5. Save artifacts: model pickle, both R90 lookups (parquet + meta.json each), the
   calibration frames themselves (val rows + predictions + error_km, for later
   inspection), and a metadata.json summarizing the run.
"""

import argparse
import json
import pickle
from datetime import datetime

import pandas as pd
from lightgbm import LGBMRegressor

from vessel_tracker.baseline import predict_baseline
from vessel_tracker.calibration import fit_conditional_radius
from vessel_tracker.config import settings
from vessel_tracker.evaluation import attach_error_km, fit_model, fit_predict
from vessel_tracker.features import build_long_dataset, model_feature_column_names
from vessel_tracker.paths import (
    CALIB_PREDICTIONS_BASELINE_PATH,
    CALIB_PREDICTIONS_LGBM_PATH,
    DATA_PROCESSED,
    LGBM_FINAL_METADATA_PATH,
    LGBM_FINAL_PATH,
    MODELS_DIR,
    PREDICTIONS_DIR,
    R90_LOOKUP_BASELINE_PATH,
    R90_LOOKUP_LGBM_PATH,
)


def _lgbm_estimator() -> LGBMRegressor:
    """create LGBM regression with set random seed"""
    return LGBMRegressor(verbose=-1, random_state=settings.random_seed, n_jobs=-1)


def _build_calibration_frame(
    trainval_long: pd.DataFrame,
    train_mmsi: set[int],
    feature_cols: list[str],
):
    """Starts with a train + val merged df in long format.
    Fit on train MMSIs, predict on val MMSIs, return val rows with error_km."""
    is_train = trainval_long["MMSI"].isin(train_mmsi)
    calib_out = fit_predict(
        trainval_long.loc[is_train, feature_cols],  # = X_train
        trainval_long.loc[is_train, ["vx", "vy"]],  # = y_train
        trainval_long.loc[~is_train, feature_cols],  # = X_pred
        _lgbm_estimator(),
    )  # returns, among other, vx_pred and vy_pred predicted ON VALIDATION SET

    # build calibration dataset, with y_true, y_pred, error etc)
    calib = trainval_long.loc[
        ~is_train,
        ["MMSI", "h", "LAT_lag_0", "LON_lag_0", "vx", "vy", "effective_speed_knots"],
    ].copy()  # features from val set for calibration
    calib = calib.rename(columns={"vx": "vx_true", "vy": "vy_true"})
    calib["vx_pred"] = calib_out["vx_pred"]
    calib["vy_pred"] = calib_out["vy_pred"]
    return attach_error_km(calib)  # add haversine error column


def _build_baseline_calibration_frame(
    trainval_long: pd.DataFrame,
    train_mmsi: set[int],
):
    """Same held-out val rows as _build_calibration_frame, no fit needed:
    predict_baseline is a closed-form constant-velocity extrapolation."""
    is_train = trainval_long["MMSI"].isin(train_mmsi)
    val_df = trainval_long.loc[~is_train]
    baseline_out = predict_baseline(val_df)

    calib = val_df[
        ["MMSI", "h", "LAT_lag_0", "LON_lag_0", "vx", "vy", "effective_speed_knots"]
    ].copy()
    calib = calib.rename(columns={"vx": "vx_true", "vy": "vy_true"})
    calib["vx_pred"] = baseline_out["vx_pred"]
    calib["vy_pred"] = baseline_out["vy_pred"]
    return attach_error_km(calib)


def train(force: bool = False) -> None:
    # ---- SKIP IF ALREADY TRAINED ----
    if (
        not force
        and LGBM_FINAL_PATH.exists()
        and LGBM_FINAL_METADATA_PATH.exists()
        and R90_LOOKUP_LGBM_PATH.exists()
        and R90_LOOKUP_BASELINE_PATH.exists()
    ):
        print(f"[skip] Artifacts already present in {MODELS_DIR} (use --force to retrain)")
        return

    # ---- LOAD DATA & BUILD FEATURES (TRAIN + VAL) ----
    print(f"Loading splits from {DATA_PROCESSED}")
    df_train = pd.read_parquet(DATA_PROCESSED / "df_train.parquet")
    df_val = pd.read_parquet(DATA_PROCESSED / "df_val.parquet")
    df_trainval = pd.concat([df_train, df_val], ignore_index=True)

    print("Building long-format features (trainval)...")
    trainval_long = build_long_dataset(df_trainval)
    feature_cols = model_feature_column_names(trainval_long.columns, include_h=True)
    print(f"  trainval: {len(trainval_long):,} rows, {trainval_long['MMSI'].nunique():,} vessels")
    print(f"  features ({len(feature_cols)}): {feature_cols}")

    # ---- FIT FINAL MODEL (served to the UI — trained on train+val) ----
    print("Fitting final LGBM on full trainval...")
    fit_out = fit_model(
        trainval_long[feature_cols],  # X_train
        trainval_long[["vx", "vy"]],  # y_train
        _lgbm_estimator(),
    )
    model = fit_out["model"]  # fitted model

    # ---- CALIBRATION: LGBM (refit on train only, residuals measured on held-out val) ----
    print("Building calibration lookups (train → val residuals)...")
    train_mmsi = set(df_train["MMSI"].unique())

    # get dataset with y_true, y_pred, h, effective speed, and haversine error of validation set
    calib_lgbm = _build_calibration_frame(trainval_long, train_mmsi, feature_cols)

    # get the marginal and conditional R(p,h,s)
    lookup_lgbm = fit_conditional_radius(calib_lgbm)  # a RadiusLookup object
    print(
        f"  lgbm calibration:     {len(calib_lgbm):,} rows, "
        f"{calib_lgbm['MMSI'].nunique():,} vessels"
    )

    # ---- CALIBRATION: BASELINE (closed-form, no fit needed) ----
    calib_baseline = _build_baseline_calibration_frame(trainval_long, train_mmsi)
    lookup_baseline = fit_conditional_radius(calib_baseline)
    print(
        f"  baseline calibration: {len(calib_baseline):,} rows, "
        f"{calib_baseline['MMSI'].nunique():,} vessels"
    )

    # ---- SAVE ARTIFACTS (model, lookups, calibration predictions, metadata) ----
    # save fitted model and lookup table for baseline and model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(LGBM_FINAL_PATH, "wb") as f:
        pickle.dump(model, f)
    lookup_lgbm.save(R90_LOOKUP_LGBM_PATH)
    lookup_baseline.save(R90_LOOKUP_BASELINE_PATH)

    # Persist the val-set calibration rows (MMSI, h, true/pred velocities, effective
    # speed, error_km)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    calib_lgbm.to_parquet(CALIB_PREDICTIONS_LGBM_PATH, index=False)
    calib_baseline.to_parquet(CALIB_PREDICTIONS_BASELINE_PATH, index=False)

    metadata = {
        "feature_cols": feature_cols,
        "lookback_minutes": settings.lookback_minutes,
        "horizon_grid_minutes": settings.horizon_grid_minutes,
        "random_seed": settings.random_seed,
        "resample_interval_min": settings.resample_interval_min,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "n_trainval_rows": int(len(trainval_long)),
        "n_trainval_vessels": int(trainval_long["MMSI"].nunique()),
        "n_calibration_rows": int(len(calib_lgbm)),
        "n_calibration_vessels": int(calib_lgbm["MMSI"].nunique()),
        "fit_time_s": float(fit_out["fit_time_s"]),
        "artifacts": {
            "model": str(LGBM_FINAL_PATH.relative_to(LGBM_FINAL_PATH.parents[2])),
            "r90_lookup_lgbm": str(
                R90_LOOKUP_LGBM_PATH.relative_to(R90_LOOKUP_LGBM_PATH.parents[2])
            ),
            "r90_lookup_baseline": str(
                R90_LOOKUP_BASELINE_PATH.relative_to(R90_LOOKUP_BASELINE_PATH.parents[2])
            ),
            "calib_predictions_lgbm": str(
                CALIB_PREDICTIONS_LGBM_PATH.relative_to(CALIB_PREDICTIONS_LGBM_PATH.parents[2])
            ),
            "calib_predictions_baseline": str(
                CALIB_PREDICTIONS_BASELINE_PATH.relative_to(
                    CALIB_PREDICTIONS_BASELINE_PATH.parents[2]
                )
            ),
        },
    }
    with open(LGBM_FINAL_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved model             → {LGBM_FINAL_PATH}")
    print(f"Saved lgbm lookup       → {R90_LOOKUP_LGBM_PATH}")
    print(f"Saved baseline lookup   → {R90_LOOKUP_BASELINE_PATH}")
    print(f"Saved lgbm predictions  → {CALIB_PREDICTIONS_LGBM_PATH}")
    print(f"Saved baseline preds    → {CALIB_PREDICTIONS_BASELINE_PATH}")
    print(f"Saved meta              → {LGBM_FINAL_METADATA_PATH}")


def main():
    parser = argparse.ArgumentParser(
        description="Train universal LGBM and R90 calibration lookups (LGBM + baseline)."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-train even if artifacts already exist",
    )
    args = parser.parse_args()
    train(force=args.force)


if __name__ == "__main__":
    main()
