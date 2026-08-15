"""Project paths — independent of current working directory."""

from pathlib import Path

# Ce fichier vit dans src/vessel_tracker/paths.py
# parents[2] remonte 2 fois : src/vessel_tracker → src → racine repo
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

CONFIGS_DIR: Path = PROJECT_ROOT / "configs"
DATA_DIR: Path = PROJECT_ROOT / "data"
DATA_RAW: Path = DATA_DIR / "raw"
DATA_PROCESSED: Path = DATA_DIR / "processed"
ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"
MODELS_DIR: Path = ARTIFACTS_DIR / "models"
PREDICTIONS_DIR: Path = ARTIFACTS_DIR / "predictions"
REPORTS_DIR: Path = ARTIFACTS_DIR / "reports"
LGBM_FINAL_PATH: Path = MODELS_DIR / "lgbm_final.pkl"
LGBM_FINAL_METADATA_PATH: Path = MODELS_DIR / "lgbm_final.metadata.json"
R90_LOOKUP_LGBM_PATH: Path = MODELS_DIR / "r90_lookup_lgbm.parquet"
R90_LOOKUP_BASELINE_PATH: Path = MODELS_DIR / "r90_lookup_baseline.parquet"
CALIB_PREDICTIONS_LGBM_PATH: Path = PREDICTIONS_DIR / "calib_predictions_lgbm.parquet"
CALIB_PREDICTIONS_BASELINE_PATH: Path = PREDICTIONS_DIR / "calib_predictions_baseline.parquet"
TEST_PREDICTIONS_PATH: Path = PREDICTIONS_DIR / "test_predictions.parquet"
TEST_REPORT_PATH: Path = REPORTS_DIR / "test_report.json"
TEST_REPORT_SUMMARY_PATH: Path = REPORTS_DIR / "test_report_summary.md"
R90_CONDITIONAL_PLOT_PATH: Path = REPORTS_DIR / "r90_conditional_vs_speed.png"
NOTEBOOK_OUTPUTS_DIR: Path = PROJECT_ROOT / "notebooks" / "outputs"
