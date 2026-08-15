"""Tests for scripts/evaluate_test.py plumbing (no full LGBM training)."""

import json

import pandas as pd
from sklearn.dummy import DummyRegressor

import evaluate_test
import train


def test_evaluate_writes_report_structure(mini_ais_df, tmp_path, monkeypatch):
    monkeypatch.setattr(train, "_lgbm_estimator", lambda: DummyRegressor())

    processed = tmp_path / "processed"
    models = tmp_path / "models"
    predictions = tmp_path / "predictions"
    reports = tmp_path / "reports"
    processed.mkdir()
    models.mkdir()

    train_df = mini_ais_df[mini_ais_df["MMSI"] == 100001]
    val_df = mini_ais_df[mini_ais_df["MMSI"] == 100002]
    train_df.to_parquet(processed / "df_train.parquet")
    val_df.to_parquet(processed / "df_val.parquet")
    val_df.to_parquet(processed / "df_test.parquet")

    lgbm_path = models / "lgbm_final.pkl"
    lookup_lgbm_path = models / "r90_lookup_lgbm.parquet"
    lookup_baseline_path = models / "r90_lookup_baseline.parquet"

    monkeypatch.setattr(train, "DATA_PROCESSED", processed)
    monkeypatch.setattr(train, "MODELS_DIR", models)
    monkeypatch.setattr(train, "PREDICTIONS_DIR", predictions)
    monkeypatch.setattr(train, "LGBM_FINAL_PATH", lgbm_path)
    monkeypatch.setattr(train, "LGBM_FINAL_METADATA_PATH", models / "lgbm_final.metadata.json")
    monkeypatch.setattr(train, "R90_LOOKUP_LGBM_PATH", lookup_lgbm_path)
    monkeypatch.setattr(train, "R90_LOOKUP_BASELINE_PATH", lookup_baseline_path)
    monkeypatch.setattr(
        train, "CALIB_PREDICTIONS_LGBM_PATH", predictions / "calib_predictions_lgbm.parquet"
    )
    monkeypatch.setattr(
        train,
        "CALIB_PREDICTIONS_BASELINE_PATH",
        predictions / "calib_predictions_baseline.parquet",
    )

    test_predictions_path = predictions / "test_predictions.parquet"
    summary_path = reports / "test_report_summary.md"
    plot_path = reports / "r90_conditional_vs_speed.png"

    monkeypatch.setattr(evaluate_test, "DATA_PROCESSED", processed)
    monkeypatch.setattr(evaluate_test, "LGBM_FINAL_PATH", lgbm_path)
    monkeypatch.setattr(evaluate_test, "R90_LOOKUP_LGBM_PATH", lookup_lgbm_path)
    monkeypatch.setattr(evaluate_test, "R90_LOOKUP_BASELINE_PATH", lookup_baseline_path)
    monkeypatch.setattr(evaluate_test, "PREDICTIONS_DIR", predictions)
    monkeypatch.setattr(evaluate_test, "TEST_PREDICTIONS_PATH", test_predictions_path)
    monkeypatch.setattr(evaluate_test, "REPORTS_DIR", reports)
    monkeypatch.setattr(evaluate_test, "TEST_REPORT_PATH", reports / "test_report.json")
    monkeypatch.setattr(evaluate_test, "TEST_REPORT_SUMMARY_PATH", summary_path)
    monkeypatch.setattr(evaluate_test, "R90_CONDITIONAL_PLOT_PATH", plot_path)

    train.train(force=True)
    evaluate_test.evaluate(force=True)

    report_path = reports / "test_report.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text())

    expected_types = {
        "created_at": str,
        "n_test_rows": int,
        "n_test_vessels": int,
        "lgbm_test_inference_time_s": float,
        "baseline_test_inference_time_s": float,
        "metrics_by_horizon": list,
        "coverage_lgbm": list,
        "coverage_baseline": list,
        "search_area_reduction_pct_by_horizon": list,
        "artifacts": dict,
    }
    for key, typ in expected_types.items():
        assert key in report
        assert isinstance(report[key], typ)

    model_names = {row["model_name"] for row in report["metrics_by_horizon"]}
    assert model_names == {"lgbm", "baseline"}

    assert len(report["search_area_reduction_pct_by_horizon"]) > 0
    for row in report["search_area_reduction_pct_by_horizon"]:
        assert {"h", "r90_lgbm_km", "r90_baseline_km", "reduction_pct"}.issubset(row)

    assert test_predictions_path.exists()
    test_predictions = pd.read_parquet(test_predictions_path)
    expected_columns = {
        "MMSI",
        "h",
        "LAT_lag_0",
        "LON_lag_0",
        "effective_speed_knots",
        "model_name",
        "vx_true",
        "vy_true",
        "vx_pred",
        "vy_pred",
        "error_km",
    }
    assert expected_columns.issubset(test_predictions.columns)
    assert set(test_predictions["model_name"].unique()) == {"lgbm", "baseline"}
    assert "test_predictions" in report["artifacts"]
    assert "summary_report" in report["artifacts"]
    assert "r90_conditional_plot" in report["artifacts"]

    assert plot_path.exists()
    assert plot_path.stat().st_size > 0

    assert summary_path.exists()
    summary_text = summary_path.read_text()
    assert "# Test-set evaluation summary" in summary_text
    assert "MAE lgbm (km)" in summary_text
    assert "MAE baseline (km)" in summary_text
    assert plot_path.name in summary_text


def test_evaluate_skips_when_report_exists(tmp_path, capsys):
    reports = tmp_path / "reports"
    reports.mkdir()
    report_path = reports / "test_report.json"
    report_path.write_text("{}")

    import evaluate_test as module

    old_path = module.TEST_REPORT_PATH
    module.TEST_REPORT_PATH = report_path
    try:
        module.evaluate(force=False)
    finally:
        module.TEST_REPORT_PATH = old_path

    assert "[skip]" in capsys.readouterr().out
