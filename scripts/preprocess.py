"""Preprocess pipeline: ingest → clean → resample → minimal filter → split."""

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from vessel_tracker.config import settings
from vessel_tracker.paths import DATA_PROCESSED, DATA_RAW
from vessel_tracker.preprocessing import (
    clean_data,
    download_source_files,
    resample_pings,
    vessel_train_test_split,
)


def expected_raw_filename() -> str:
    """Build expected raw merged filename from ingestion config.

    Must match the convention used by download_source_files().
    """
    ing = settings.ingestion
    bb = ing.bounding_box
    coords = f"lon{bb.lon_west:.1f}to{bb.lon_east:.1f}_lat{bb.lat_south:.1f}to{bb.lat_north:.1f}"
    start = datetime.strptime(ing.start_date, "%Y-%m-%d").strftime("%Y%m%d")
    end = datetime.strptime(ing.end_date, "%Y-%m-%d").strftime("%Y%m%d")
    return f"AIS_merged_{coords}_{start}to{end}.parquet"


def ingest(force: bool = False) -> Path:
    """Download raw AIS data if not already present. Returns path to raw file."""
    raw_path = DATA_RAW / expected_raw_filename()
    if raw_path.exists() and not force:
        print(f"[skip ingest] Raw file already present: {raw_path.name}")
        return raw_path

    print("[ingest] Downloading from NOAA...")
    print(f"  Period:       {settings.ingestion.start_date} → {settings.ingestion.end_date}")
    print(
        f"  Bounding box: lon [{settings.ingestion.bounding_box.lon_west}, "
        f"{settings.ingestion.bounding_box.lon_east}], "
        f"lat [{settings.ingestion.bounding_box.lat_south}, "
        f"{settings.ingestion.bounding_box.lat_north}]"
    )

    DATA_RAW.parent.mkdir(parents=True, exist_ok=True)
    download_source_files(
        start_date=settings.ingestion.start_date,
        end_date=settings.ingestion.end_date,
        lon_west=settings.ingestion.bounding_box.lon_west,
        lon_east=settings.ingestion.bounding_box.lon_east,
        lat_north=settings.ingestion.bounding_box.lat_north,
        lat_south=settings.ingestion.bounding_box.lat_south,
        output_path=str(DATA_RAW.parent),
    )  # data are organized as data/temp and data/raw,
    # so, we need to launch download_source_files at the data/ level

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Expected raw file {raw_path} was not created by download_source_files. "
            "Check the function output."
        )
    print(f"[ingest] Downloaded to {raw_path}")
    return raw_path


def should_skip_processing(raw_path: Path, force: bool) -> bool:
    """Skip clean+resample+split if outputs exist and are newer than raw."""
    if force:
        return False
    resampled_name = f"AIS_clean_resampled_{settings.resample_interval_min}min.parquet"
    required = [
        resampled_name,
        "df_train.parquet",
        "df_val.parquet",
        "df_test.parquet",
    ]
    paths = [DATA_PROCESSED / f for f in required]
    if not all(p.exists() for p in paths):
        return False
    raw_mtime = raw_path.stat().st_mtime
    return all(p.stat().st_mtime >= raw_mtime for p in paths)


def apply_minimal_eligibility_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Drop vessels with fewer than 2h of resampled track."""
    min_steps = int(2 * 60 / settings.resample_interval_min)
    print(
        f"[filter] Minimal eligibility: ≥2h ({min_steps} steps @ "
        f"{settings.resample_interval_min} min)"
    )
    track_len = df.groupby("MMSI").size()
    eligible_mmsi = track_len[track_len >= min_steps].index
    n_excluded = df["MMSI"].nunique() - len(eligible_mmsi)
    print(f"  → {len(eligible_mmsi):,} / {len(track_len):,} vessels kept ({n_excluded} excluded)")
    return df[df["MMSI"].isin(eligible_mmsi)].copy()


def process(raw_path: Path):
    """Clean, resample, filter, and split the raw merged parquet."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)  # data/processed by default

    print(f"[load] Reading {raw_path.name}")
    df = pd.read_parquet(raw_path)
    df["BaseDateTime"] = pd.to_datetime(df["BaseDateTime"])
    print(f"  → {len(df):,} rows, {df['MMSI'].nunique():,} unique MMSI")

    print("[clean] clean_data")
    df = clean_data(df)
    print(f"  → {len(df):,} rows after cleaning")

    print(f"[resample] resample_pings (interval={settings.resample_interval_min}min)")
    df = resample_pings(df)  # to settings.resample_interval_min by default
    print(f"  → {len(df):,} rows after resampling")

    df = apply_minimal_eligibility_filter(df)
    print(f"  → {len(df):,} rows, {df['MMSI'].nunique():,} vessels after filter")

    resampled_path = (
        DATA_PROCESSED / f"AIS_clean_resampled_{settings.resample_interval_min}min.parquet"
    )
    df.to_parquet(resampled_path, index=False)
    print(f"[save] {resampled_path.name} ({len(df):,} rows, {df['MMSI'].nunique():,} vessels)")

    print("[split] vessel_train_test_split (groupé par MMSI, ratios depuis YAML)")
    # also returns groups, not used here
    df_train, df_val, df_test, *_ = vessel_train_test_split(
        df,
        test_size=settings.split_ratios.test,
        val_size=settings.split_ratios.val,
    )
    print(f"  Train: {len(df_train):,} rows, {df_train['MMSI'].nunique():,} vessels")
    print(f"  Val:   {len(df_val):,} rows, {df_val['MMSI'].nunique():,} vessels")
    print(f"  Test:  {len(df_test):,} rows, {df_test['MMSI'].nunique():,} vessels")

    df_train.to_parquet(DATA_PROCESSED / "df_train.parquet")
    df_val.to_parquet(DATA_PROCESSED / "df_val.parquet")
    df_test.to_parquet(DATA_PROCESSED / "df_test.parquet")
    print(f"[done] Written to {DATA_PROCESSED}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess AIS data: ingest, clean, resample, split."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-processing of clean/resample/split. Does NOT re-download raw data. "
        "To re-download, delete data/raw/AIS_merged_*.parquet manually.",
    )
    args = parser.parse_args()

    raw_path = ingest(force=False)  # skipped if ingestion already done and return path to raw data
    if should_skip_processing(
        raw_path, force=args.force
    ):  # skip preprocess if done and force = False
        print(f"[skip process] Processed files already up-to-date in {DATA_PROCESSED}")
        return
    process(raw_path)


if __name__ == "__main__":
    main()
