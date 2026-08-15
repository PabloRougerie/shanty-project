"""Conditional uncertainty radius R(p, h, s) from calibration-set residuals."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def _horizon_hours(h_minutes: pd.Series) -> pd.Series:
    """Convert a forecast horizon from minutes (raw `h` column) to whole hours."""
    return (h_minutes / 60).round().astype(int)


def _speed_bin_edges(speeds: pd.Series, n_bins: int) -> np.ndarray:
    """Compute quantile-based speed bin edges, with outer bounds opened to +/- inf.

    Args:
        speeds: Calibration-set effective speeds (knots) used to fit the edges.
        n_bins: Target number of quantile bins (may yield fewer, see note below).

    Returns:
        Sorted 1D array of edges (length <= n_bins + 1). The first and last edges
        are forced to -inf/+inf so any future speed (e.g. from the test set) falls
        into the first or last bin instead of being dropped as out-of-range.
    """
    # Split the effective-speed series into quantile bins (equal count per bin, width may vary).
    # duplicates="drop" avoids a ValueError when repeated speed values create duplicate edges
    # (e.g. many vessels at 0 kn), at the cost of possibly returning fewer than n_bins edges.
    _, edges = pd.qcut(speeds, n_bins, retbins=True, duplicates="drop")
    edges = edges.astype(float).copy()

    # Open the outer boundaries so speeds outside the calibration range (e.g. in the test set)
    # still fall into the first/last bin instead of being dropped as NaN.
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def _assign_speed_bin(speeds: pd.Series, edges: np.ndarray) -> pd.Series:
    """Map each speed to its bin index using pre-computed edges.

    Args:
        speeds: Effective speeds (knots) to bin (calibration or test data).
        edges: Bin edges, as returned by `_speed_bin_edges`.

    Returns:
        Series (same index as `speeds`) of 0-based integer bin indices.
    Notes:
        ANy value outside the edge range becomes NaN, which upcasts the whole Series to float.
    """
    return pd.cut(speeds, bins=edges, labels=False)


@dataclass
class RadiusLookup:
    """Conditional radius table (per speed bin and horizon), plus a marginal
    (speed-independent) fallback table for the same horizons."""

    percentile: float
    speed_bin_edges: np.ndarray
    lookup_df: pd.DataFrame
    marginal_df: pd.Series

    def save(self, path: str | Path) -> None:
        """Persist the lookup as a parquet table plus a `.meta.json` sidecar."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        lookup_to_save = self.lookup_df.copy()
        lookup_to_save["percentile"] = self.percentile
        lookup_to_save.to_parquet(path, index=False)
        meta = {
            "percentile": self.percentile,
            "speed_bin_edges": self.speed_bin_edges.tolist(),
            # reset_index() turns the Series (indexed by horizon_hours) into a
            # two-column frame so to_dict(orient="records") can serialize it.
            "marginal": self.marginal_df.reset_index().to_dict(orient="records"),
        }
        # with_suffix replaces the ".parquet" extension with ".meta.json" (same stem, sidecar file).
        path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> RadiusLookup:
        """Reconstruct a RadiusLookup from the parquet + `.meta.json` pair written by `save`."""
        path = Path(path)
        loaded_parquet = pd.read_parquet(path)
        percentile = float(loaded_parquet["percentile"].iloc[0])
        lookup_df = loaded_parquet.drop(columns=["percentile"])
        meta = json.loads(path.with_suffix(".meta.json").read_text())
        marginal_records = meta["marginal"]
        marginal_df = pd.DataFrame(marginal_records).set_index("horizon_hours")["radius_km"]
        return cls(
            percentile=percentile,
            speed_bin_edges=np.asarray(meta["speed_bin_edges"], dtype=float),
            lookup_df=lookup_df,
            marginal_df=marginal_df,
        )


def fit_conditional_radius(
    calibration_df: pd.DataFrame,
    *,
    percentile: float = 0.90,
    n_speed_bins: int = 10,
    speed_col: str = "effective_speed_knots",
    error_col: str = "error_km",
) -> RadiusLookup:
    """Fit speed-conditioned radius lookup on a held-out calibration split.

    Args:
        calibration_df: Val rows with h, effective_speed_knots, and error_km
            already computed upstream.
        percentile: Error quantile for the radius (0.90 = R90).
        n_speed_bins: Number of speed quantile bins (qcut on calibration set).
        speed_col: Effective speed column in knots.
        error_col: Residual column name in km.

    Returns:
        RadiusLookup with conditional and marginal radii per horizon (hours).

    Note:
        `error_km` must already be attached on `calibration_df` before calling this
        (e.g. via `_build_calibration_frame` or `_build_baseline_calibration_frame`).
    """
    required = ["h", speed_col, error_col]
    missing = [col for col in required if col not in calibration_df.columns]
    if missing:
        raise ValueError(f"Missing required calibration columns: {missing}")

    calib = calibration_df.copy()

    calib["horizon_hours"] = _horizon_hours(calib["h"])
    edges = _speed_bin_edges(calib[speed_col], n_speed_bins)  # defin edges of calibration set
    calib["speed_bin_index"] = _assign_speed_bin(
        calib[speed_col], edges
    )  # assign each speed of calib set to a specific bin

    # Edge definition and bin assignment are kept as separate helpers
    # so `_assign_speed_bin` can be reused as-is.

    # For each (speed_bin, horizon) group, take the `percentile`-th quantile of the
    # haversine errors of calibration rows that fall in that group -> one radius_km per group.
    lookup = (
        calib.groupby(["speed_bin_index", "horizon_hours"], observed=True)[error_col]
        .quantile(percentile)
        .rename("radius_km")
        .reset_index()
    )
    marginal = calib.groupby("horizon_hours")[error_col].quantile(percentile).rename("radius_km")

    return RadiusLookup(
        percentile=percentile,
        speed_bin_edges=edges,
        lookup_df=lookup,
        marginal_df=marginal,
    )


def coverage_report(
    results_df: pd.DataFrame,
    lookup: RadiusLookup,
    *,
    speed_col: str = "effective_speed_knots",
    error_col: str = "error_km",
) -> pd.DataFrame:
    """Measure test-set coverage inside conditional and marginal R90 disks.

    Args:
        results_df: Test rows with effective speed, horizon and model error_km.
        lookup: RadiusLookup built on the calibration split.
        speed_col: Effective speed column in knots.
        error_col: Model haversine error column in km.

    Returns:
        One row per (speed_bin, horizon_hours) with radii and coverage percentages.
    """
    required = [speed_col, error_col, "h"]
    missing = [col for col in required if col not in results_df.columns]
    if missing:
        raise ValueError(f"Missing required results columns: {missing}")

    results = results_df.copy()
    results["horizon_hours"] = _horizon_hours(results["h"])

    # Reuse the calibration set's bin edges to see where the test set speeds fall.
    results["speed_bin_index"] = _assign_speed_bin(results[speed_col], lookup.speed_bin_edges)

    # we'll need to check coverage for each (bin, h) couple.
    # Using pd conditonal masking would take time
    # so, we pre-index a dict-based lookup table
    # that already stores the (bin,h) couple and their corresponding R.

    lookup_map = {
        (int(r.speed_bin_index), int(r.horizon_hours)): float(r.radius_km)
        for r in lookup.lookup_df.itertuples(index=False)
    }

    rows = []
    grouped = results.groupby(["speed_bin_index", "horizon_hours"], observed=True)
    for (bin_idx, horizon), grp in grouped:
        if pd.isna(bin_idx):
            # Test speed fell outside the calibration speed range; skip (should not
            # happen since edges are opened to +/- inf, kept as a defensive guard).
            continue
        key = (int(bin_idx), int(horizon))
        r_cond = lookup_map.get(key)
        if r_cond is None:
            # No calibration rows for this (bin, horizon) pair, so no conditional
            # radius was fitted for it; nothing to compare coverage against.
            continue
        r_marg = float(lookup.marginal_df.loc[horizon])
        rows.append(
            {
                "speed_bin_index": int(bin_idx),
                "horizon_hours": int(horizon),
                "n_test": len(grp),
                "r90_conditional_km": r_cond,
                "coverage_conditional_pct": (grp[error_col] <= r_cond).mean() * 100,
                "r90_marginal_km": r_marg,
                "coverage_marginal_pct": (grp[error_col] <= r_marg).mean() * 100,
            }
        )

    return pd.DataFrame(rows)
