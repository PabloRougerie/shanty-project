"""Shared synthetic fixtures for package and UI tests."""

import pandas as pd
import pytest


def _make_vessel_track(mmsi: int, n_pings: int, lat0: float, lon0: float) -> pd.DataFrame:
    """Build one vessel track: n_pings rows spaced 10 min apart."""
    times = pd.date_range("2024-11-01 08:00", periods=n_pings, freq="10min")
    rows = []
    for i, t in enumerate(times):
        rows.append(
            {
                "MMSI": mmsi,
                "BaseDateTime": t,
                "LAT": lat0 + i * 0.01,
                "LON": lon0 + i * 0.01,
                "SOG": 10.0,
                "COG": 45.0,
                "Heading": 45.0,
                "Status": 0.0,
                "Length": 200.0,
                "Width": 30.0,
                "Draft": 10.0,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def mini_ais_df() -> pd.DataFrame:
    """Two vessels, 120 pings each (10 min spacing). Enough for 12h lookback tests."""
    df1 = _make_vessel_track(mmsi=100001, n_pings=120, lat0=48.0, lon0=-5.0)
    df2 = _make_vessel_track(mmsi=100002, n_pings=120, lat0=40.0, lon0=-3.0)
    return pd.concat([df1, df2], ignore_index=True)


@pytest.fixture
def multi_vessel_df() -> pd.DataFrame:
    """Three vessels for split tests (enough groups for train/val/test)."""
    frames = [
        _make_vessel_track(mmsi=100001, n_pings=30, lat0=48.0, lon0=-5.0),
        _make_vessel_track(mmsi=100002, n_pings=30, lat0=40.0, lon0=-3.0),
        _make_vessel_track(mmsi=100003, n_pings=30, lat0=35.0, lon0=-8.0),
    ]
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def vessel_logs_varied_lengths() -> pd.DataFrame:
    """Two vessels: one below min_track_steps, one long enough for demo export."""
    from vessel_tracker.config import settings

    min_pings = settings.min_track_steps
    short = _make_vessel_track(mmsi=111111, n_pings=min_pings - 1, lat0=48.0, lon0=-5.0)
    long = _make_vessel_track(mmsi=222222, n_pings=min_pings + 10, lat0=40.0, lon0=-3.0)
    return pd.concat([short, long], ignore_index=True)
