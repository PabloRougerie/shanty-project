"""Score LGBM and the constant-velocity baseline on the held-out test set.

Pure reporting: loads what scripts/train.py produced (model + both R90 lookups),
predicts vx/vy for the test, and checks R90 coverage: which
% of test-set haversine errors fall below the R90 radius (good calibration means
this % is close to 90).

Steps performed by evaluate():
1. Load df_test, build long-format features.
2. Load the artifacts produced by train.py: the pickled LGBM model and both R90
   lookups (lgbm + baseline).
3. Predict vx/vy on the test set with both models, attach haversine error_km.
4. Compute MAE/R90 metrics per horizon and per model (compute_metrics), and the
   search-area reduction of lgbm vs baseline.
5. Check R90 calibration coverage: % of test rows whose error_km falls under the
   R90 radius predicted by each lookup (target ~90%).
6. Save artifacts: the combined per-row test predictions (model_name, true/pred
   velocities, error_km), a metadata.json-style report with all metrics above, and
   a human-readable summary (Markdown table + plot) for non-technical readers.
"""

import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from vessel_tracker.baseline import predict_baseline
from vessel_tracker.calibration import RadiusLookup, coverage_report
from vessel_tracker.evaluation import attach_error_km, predict_model
from vessel_tracker.features import build_long_dataset, model_feature_column_names
from vessel_tracker.metrics import compute_metrics
from vessel_tracker.paths import (
    DATA_PROCESSED,
    LGBM_FINAL_PATH,
    PREDICTIONS_DIR,
    R90_CONDITIONAL_PLOT_PATH,
    R90_LOOKUP_BASELINE_PATH,
    R90_LOOKUP_LGBM_PATH,
    REPORTS_DIR,
    TEST_PREDICTIONS_PATH,
    TEST_REPORT_PATH,
    TEST_REPORT_SUMMARY_PATH,
)

_RESULT_COLUMNS = ["MMSI", "h", "LAT_lag_0", "LON_lag_0", "vx", "vy", "effective_speed_knots"]


def _predict_results(model_name: str, test_long: pd.DataFrame, pred_out: dict) -> pd.DataFrame:
    """Assemble the (model_name, error_km) results frame from a predict_*-style dict."""
    results = test_long[_RESULT_COLUMNS].copy()
    results = results.rename(columns={"vx": "vx_true", "vy": "vy_true"})
    results["model_name"] = model_name
    results["vx_pred"] = pred_out["vx_pred"]
    results["vy_pred"] = pred_out["vy_pred"]
    return attach_error_km(results)


def _search_area_reduction(metrics_by_horizon: pd.DataFrame) -> list[dict]:
    """(1 - (r90_lgbm / r90_baseline)**2) * 100 per horizon, from test-set R90 (NB07 metric)."""
    pivot = metrics_by_horizon.pivot(index="h", columns="model_name", values="r90_km")
    rows = []
    for h, row in pivot.iterrows():
        r90_lgbm = float(row["lgbm"])
        r90_baseline = float(row["baseline"])
        rows.append(
            {
                "h": int(h),
                "r90_lgbm_km": r90_lgbm,
                "r90_baseline_km": r90_baseline,
                "reduction_pct": float((1 - (r90_lgbm / r90_baseline) ** 2) * 100),
            }
        )
    return rows


def _build_summary_table(
    metrics_by_horizon: pd.DataFrame, search_area_reduction: list[dict]
) -> pd.DataFrame:
    """One row per horizon: MAE and marginal R90 for both models, plus lgbm's search-area
    reduction vs baseline (from `_search_area_reduction`)."""
    mae_pivot = metrics_by_horizon.pivot(index="h", columns="model_name", values="mae_mean")
    reduction_df = pd.DataFrame(search_area_reduction).set_index("h")
    table = pd.DataFrame(
        {
            "h (min)": mae_pivot.index,
            "MAE lgbm (km)": mae_pivot["lgbm"].round(1),
            "MAE baseline (km)": mae_pivot["baseline"].round(1),
            "R90 lgbm (km)": reduction_df["r90_lgbm_km"].round(1),
            "R90 baseline (km)": reduction_df["r90_baseline_km"].round(1),
            "Search-area reduction": reduction_df["reduction_pct"].round(1).astype(str) + "%",
        }
    ).reset_index(drop=True)
    return table


def _plot_r90_conditional_vs_speed(
    coverage_lgbm: pd.DataFrame, coverage_baseline: pd.DataFrame, plot_path: Path
) -> None:
    """Side-by-side (lgbm | baseline) plot of the conditional R90 radius against the
    effective-speed decile, one line per horizon. Mirrors the NB07 coverage plot style
    (visualizations/Rmarginal_vs_conditional.png), but for the radius itself, model vs model.
    """
    horizons_hours = sorted(coverage_lgbm["horizon_hours"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, df, title in zip(axes, [coverage_lgbm, coverage_baseline], ["LGBM", "Baseline"]):
        for h in horizons_hours:
            subset = df[df["horizon_hours"] == h].sort_values("speed_bin_index")
            ax.plot(
                subset["speed_bin_index"],
                subset["r90_conditional_km"],
                marker="o",
                label=f"{h}h",
            )
        ax.set_title(title)
        ax.set_xlabel("Effective speed decile (low → high)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Conditional R90 radius (km)")
    axes[1].legend(title="Horizon", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle("Conditional R90 radius vs effective speed, LGBM vs baseline")
    fig.tight_layout()

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _write_summary_markdown(
    summary_table: pd.DataFrame, plot_path: Path, md_path: Path, created_at: str
) -> None:
    """Human-readable Markdown report: MAE/R90 summary table + conditional-R90 plot."""
    lines = [
        "# Test-set evaluation summary",
        "",
        f"Generated: {created_at}",
        "",
        "## Mean error and marginal R90, by horizon",
        "",
        summary_table.to_markdown(index=False),
        "",
        "## Conditional R90 radius vs effective speed",
        "",
        f"![Conditional R90 radius vs effective speed]({plot_path.name})",
        "",
    ]
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines))


def evaluate(force: bool = False) -> None:
    # ---- SKIP IF ALREADY EVALUATED ----
    if not force and TEST_REPORT_PATH.exists():
        print(f"[skip] Report already present at {TEST_REPORT_PATH} (use --force to rebuild)")
        return

    # ---- LOAD DATA & BUILD FEATURES (TEST) ----
    print(f"Loading test split from {DATA_PROCESSED}")
    df_test = pd.read_parquet(DATA_PROCESSED / "df_test.parquet")

    print("Building long-format test features...")
    test_long = build_long_dataset(df_test)
    feature_cols = model_feature_column_names(test_long.columns, include_h=True)
    print(f"  test: {len(test_long):,} rows, {test_long['MMSI'].nunique():,} vessels")

    # ---- LOAD TRAINED ARTIFACTS (model + both R90 lookups) ----
    print("Loading trained artifacts...")
    with open(LGBM_FINAL_PATH, "rb") as f:
        model = pickle.load(f)
    lookup_lgbm = RadiusLookup.load(R90_LOOKUP_LGBM_PATH)
    lookup_baseline = RadiusLookup.load(R90_LOOKUP_BASELINE_PATH)

    # ---- PREDICT ON TEST (lgbm + baseline) ----
    print("Predicting on held-out test (lgbm + baseline)...")
    lgbm_pred_out = predict_model(model, test_long[feature_cols])  # dict with vx pred and vy pred
    results_lgbm = _predict_results("lgbm", test_long, lgbm_pred_out)

    baseline_pred_out = predict_baseline(test_long)
    results_baseline = _predict_results("baseline", test_long, baseline_pred_out)

    combined = pd.concat([results_lgbm, results_baseline], ignore_index=True)
    metrics_by_horizon = compute_metrics(combined)  # get the mean MAE, and different R(p,h)

    # ---- CHECK R90 CALIBRATION COVERAGE ----
    print("Checking R90 calibration coverage...")
    coverage_lgbm = coverage_report(results_lgbm, lookup_lgbm)
    coverage_baseline = coverage_report(results_baseline, lookup_baseline)

    # ---- SAVE ARTIFACTS (test predictions + report + human-readable summary) ----
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(TEST_PREDICTIONS_PATH, index=False)

    created_at = datetime.now().isoformat(timespec="seconds")
    search_area_reduction = _search_area_reduction(metrics_by_horizon)

    report = {
        "created_at": created_at,
        "n_test_rows": int(len(test_long)),
        "n_test_vessels": int(test_long["MMSI"].nunique()),
        "lgbm_test_inference_time_s": float(lgbm_pred_out["inference_time_s"]),
        "baseline_test_inference_time_s": float(baseline_pred_out["inference_time_s"]),
        "metrics_by_horizon": metrics_by_horizon.to_dict(orient="records"),
        "coverage_lgbm": coverage_lgbm.to_dict(orient="records"),
        "coverage_baseline": coverage_baseline.to_dict(orient="records"),
        "search_area_reduction_pct_by_horizon": search_area_reduction,
        "artifacts": {
            "test_predictions": str(
                TEST_PREDICTIONS_PATH.relative_to(TEST_PREDICTIONS_PATH.parents[2])
            ),
            "summary_report": str(
                TEST_REPORT_SUMMARY_PATH.relative_to(TEST_REPORT_SUMMARY_PATH.parents[2])
            ),
            "r90_conditional_plot": str(
                R90_CONDITIONAL_PLOT_PATH.relative_to(R90_CONDITIONAL_PLOT_PATH.parents[2])
            ),
        },
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(TEST_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print("Building human-readable summary (table + plot)...")
    summary_table = _build_summary_table(metrics_by_horizon, search_area_reduction)
    _plot_r90_conditional_vs_speed(coverage_lgbm, coverage_baseline, R90_CONDITIONAL_PLOT_PATH)
    _write_summary_markdown(
        summary_table, R90_CONDITIONAL_PLOT_PATH, TEST_REPORT_SUMMARY_PATH, created_at
    )

    print(f"Saved test predictions → {TEST_PREDICTIONS_PATH}")
    print(f"Saved report           → {TEST_REPORT_PATH}")
    print(f"Saved summary          → {TEST_REPORT_SUMMARY_PATH}")
    print(f"Saved R90 plot          → {R90_CONDITIONAL_PLOT_PATH}")


def main():
    parser = argparse.ArgumentParser(
        description="Score LGBM and the constant-velocity baseline on the held-out test set."
    )
    parser.add_argument(
        "--force",
        action="store_true",  # force = True
        help="Re-evaluate even if the report already exists",
    )
    args = parser.parse_args()
    evaluate(force=args.force)


if __name__ == "__main__":
    main()
