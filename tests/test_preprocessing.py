"""Tests for preprocessing splits."""

import numpy as np
import pytest

from vessel_tracker.preprocessing import vessel_train_test_split


def test_vessel_train_test_split_no_mmsi_leak(multi_vessel_df):
    """Nominal case: no MMSI appears in more than one split."""
    df_train, df_val, df_test, *_ = vessel_train_test_split(
        multi_vessel_df, test_size=0.33, val_size=0.33, random_state=0
    )
    mmsi_train = set(df_train["MMSI"].unique())
    mmsi_val = set(df_val["MMSI"].unique())
    mmsi_test = set(df_test["MMSI"].unique())

    assert len(mmsi_train) > 0
    assert len(mmsi_val) > 0
    assert len(mmsi_test) > 0
    assert mmsi_train.isdisjoint(mmsi_val)
    assert mmsi_train.isdisjoint(mmsi_test)
    assert mmsi_val.isdisjoint(mmsi_test)


def test_vessel_train_test_split_raises_on_mmsi_leak(multi_vessel_df, monkeypatch):
    """Negative case: forced overlap between train and val must raise ValueError."""

    class LeakyGroupShuffleSplit:
        """Yields overlapping indices on the second split call."""

        def __init__(self, *args, **kwargs):
            self._call = 0

        def split(self, X, y=None, groups=None):
            n = len(X)
            if self._call == 0:
                self._call += 1
                mid = n // 2
                yield np.arange(mid), np.arange(mid, n)
            else:
                idx = np.arange(n)
                yield idx, idx  # same rows in train and val → MMSI leak

    # temporary replacement of GrouShuffleSplit during the test
    monkeypatch.setattr(
        "vessel_tracker.preprocessing.GroupShuffleSplit",
        LeakyGroupShuffleSplit,
    )
    # pytest.raises catches ValueError "MMSI leakage"
    # the exception doesnt break the test, the test passes
    with pytest.raises(ValueError, match="MMSI leakage"):
        vessel_train_test_split(multi_vessel_df, random_state=0)


def test_vessel_train_test_split_missing_mmsi_raises(multi_vessel_df):
    df = multi_vessel_df.drop(columns=["MMSI"])
    with pytest.raises(ValueError, match="MMSI"):
        vessel_train_test_split(df)
