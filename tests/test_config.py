"""Config v3 alignment with NB07."""

from vessel_tracker.config import settings


def test_horizon_grid_matches_nb07():
    assert settings.horizon_grid_minutes == [120, 360, 480, 600, 720, 1440, 2880, 4320]


def test_fixed_lookback_and_lags():
    assert settings.lookback_minutes == 720
    assert settings.nb_lags == 6
    assert settings.lookback_steps() == 72


def test_model_feature_columns():
    assert settings.feature_columns == ["Length", "Width", "Draft", "dx", "dy"]


def test_min_track_steps_covers_lookback_plus_max_horizon():
    lookback_steps = settings.lookback_steps()
    max_horizon_steps = settings.horizon_steps(settings.max_horizon_minutes)
    assert settings.min_track_steps == lookback_steps + max_horizon_steps
