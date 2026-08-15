import gc
import zipfile
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import GroupShuffleSplit

from vessel_tracker.config import settings
from vessel_tracker.metrics import haversine_distance

SOG_MAX_GAP_MIN = 30


def download_source_files(
    start_date: str, end_date, lon_west, lon_east, lat_north, lat_south, output_path="../data/"
):
    """
    https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2024/AIS_2024_12_31.zip
    """

    # convert date into date time
    start_date = pd.to_datetime(start_date, format="%Y-%m-%d")
    end_date = pd.to_datetime(end_date, format="%Y-%m-%d")
    temp_dir_path = Path(output_path) / "temp"
    temp_dir_path.mkdir(parents=True, exist_ok=True)

    current_date = start_date

    while current_date <= end_date:
        year = current_date.year
        month = current_date.month
        day = current_date.day
        temp_parquet = temp_dir_path / f"temp_AIS_{year}_{month:02d}_{day:02d}.parquet"
        if temp_parquet.exists():
            print(f"Skipping {current_date.date()} — daily parquet already exists")
            current_date += timedelta(days=1)
            continue

        # 1. ZIP FILE DOWNLOAD
        # construct URL to fetch files on NOAA according to their naming convention
        url = f"https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{year}/AIS_{year}_{month:02d}_{day:02d}.zip"

        # Download zip to disk in chunks (avoids loading the full file in RAM).
        temp_zip = temp_dir_path / f"temp_AIS_{year}_{month:02d}_{day:02d}.zip"
        print(f"Requestion zip files for date {current_date}")
        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            with temp_zip.open("wb") as f:
                for chunk in response.iter_content(chunk_size=1 << 20):  # 1 MiB
                    if chunk:
                        f.write(chunk)  # write each chunk of the zip files sent by NOAA on disk

        # READ CSV FROM ZIP
        with zipfile.ZipFile(temp_zip) as zip:
            filename = zip.namelist()[0]  # get csv file name from zip files list

            # open the file from within the zipfile
            with zip.open(filename) as csv_file:
                df = pd.read_csv(csv_file)

        temp_zip.unlink(missing_ok=True)

        # FILTER GEOGRAPHIC AREA OF INTEREST
        # doing it right away to minimize file size to be saved to disk.
        # Lon are negative west of greenwhich
        print("Restricting to AOI")
        mask = (
            (df["LON"] >= lon_west)
            & (df["LON"] <= lon_east)
            & (df["LAT"] >= lat_south)
            & (df["LAT"] <= lat_north)
        )
        df_bound = df.loc[mask, :]

        # FILTER BY VESSEL TYPE
        print("Selecting cargo ships...")
        df_bound = df_bound.loc[(df_bound["VesselType"] >= 70) & (df_bound["VesselType"] < 90), :]

        # SELECT RELEVANT COLUMNS
        print("Filtering features...")
        df_filtered = df_bound.drop(
            columns=["VesselName", "IMO", "CallSign", "VesselType", "Cargo", "TransceiverClass"]
        )
        # SAVE TEMP PARQUET FILE
        df_filtered.to_parquet(temp_parquet, index=False)

        del df, df_bound, df_filtered
        gc.collect()
        current_date += timedelta(days=1)

    # find all parquet files in temp dir (sorted by filename, hence by date)
    temp_parquet_files = sorted(temp_dir_path.glob("*.parquet"))

    # security check if there is no file
    if not temp_parquet_files:
        print("No parquet files found in temp dir")
        return

    print(f"Merging {len(temp_parquet_files)} parquet files into a dataframe")
    # read all daily parquet files
    dataframes = [pd.read_parquet(file) for file in temp_parquet_files]

    # merge
    # concatenate vertically
    df_merged = pd.concat(dataframes, ignore_index=True, axis=0)
    raw_dir_path = Path(output_path) / "raw"
    raw_dir_path.mkdir(parents=True, exist_ok=True)

    # Create descriptive filename with bounding box coordinates and date range
    # Format coordinates: lon{west}to{east}_lat{south}to{north}
    # Format dates: {start_date}to{end_date} (YYYYMMDD)
    coords_str = f"lon{lon_west:.1f}to{lon_east:.1f}_lat{lat_south:.1f}to{lat_north:.1f}"
    dates_str = f"{start_date.strftime('%Y%m%d')}to{end_date.strftime('%Y%m%d')}"
    parquet_merged_filename = raw_dir_path / f"AIS_merged_{coords_str}_{dates_str}.parquet"

    df_merged.to_parquet(parquet_merged_filename, index=False)

    for path in temp_dir_path.iterdir():
        if path.is_file():
            path.unlink()
    print(f"Cleaned temporary directory {temp_dir_path}.")

    # clean memory
    del dataframes, df_merged
    gc.collect()


def interpolate_sog_gap_limited(g):
    g = g.sort_values("BaseDateTime").copy().reset_index(drop=True)
    is_nan = g["SOG"].isna()

    last_known_time = g["BaseDateTime"].where(~is_nan).ffill()
    next_known_time = g["BaseDateTime"].where(~is_nan).bfill()
    gap_duration_min = (next_known_time - last_known_time).dt.total_seconds() / 60

    g = g.set_index("BaseDateTime")
    g["SOG"] = g["SOG"].interpolate(method="time", limit_area="inside")
    g = g.reset_index()

    g.loc[is_nan & (gap_duration_min > SOG_MAX_GAP_MIN), "SOG"] = np.nan
    return g


_JUMP_SPEED_MAX_KN = 50  # well above the ~25 kn ceiling for cargo/tankers


def _add_interping_dynamics(df):
    """Per-vessel inter-ping duration (h), distance (NM), speed (kn), jump flag."""
    df = df.sort_values(["MMSI", "BaseDateTime"]).copy()
    df["interping_duration"] = df.groupby("MMSI")["BaseDateTime"].diff().dt.total_seconds() / 3600
    lat_prev = df.groupby("MMSI")["LAT"].shift(1)
    lon_prev = df.groupby("MMSI")["LON"].shift(1)
    df["interping_distance"] = (
        haversine_distance(lat_prev, lon_prev, df["LAT"], df["LON"]) / 1.852
    )  # km → NM, so speed comes out in knots
    df["interping_speed"] = (df["interping_distance"] / df["interping_duration"]).replace(
        np.inf, np.nan
    )
    df["is_jump"] = df["interping_speed"] > _JUMP_SPEED_MAX_KN
    return df


def clean_data(df):
    """Remove duplicates, impute missing values, filter GPS jumps."""

    # --- Decode sentinel-encoded missing values ---
    dimension_cols = ["Length", "Width", "Draft"]
    df[dimension_cols] = df[dimension_cols].replace(0, np.nan)
    df["SOG"] = df["SOG"].where(df["SOG"] < 102.2, np.nan)
    df["Heading"] = df["Heading"].replace(511, np.nan)
    df["COG"] = df["COG"].replace(360, np.nan)
    df["LAT"] = df["LAT"].replace(91, np.nan)
    df["LON"] = df["LON"].replace(181, np.nan)

    df = df.drop_duplicates()

    # --- Ship dimensions imputation (Length, Width, Draft) ---
    df = df.fillna({col: df[col].median() for col in dimension_cols})

    # --- SOG imputation (gaps ≤ SOG_MAX_GAP_MIN min, per vessel) ---
    df = df.sort_values(["MMSI", "BaseDateTime"])
    df = df.groupby("MMSI", group_keys=False).apply(interpolate_sog_gap_limited)

    # --- Drop COG: redundant with LAT/LON sequence, missingness concentrated by vessel ---
    df = df.drop(columns=["COG"])

    # --- GPS jump filter (two-pass) ---
    # Pass 1: drop rows flagged as jumps — clears isolated GPS spikes
    df = _add_interping_dynamics(df)
    df = df[~df["is_jump"]]

    # Pass 2: recompute on new adjacencies; drop any vessel that still jumps
    df = _add_interping_dynamics(df)
    vessels_with_residual_jump = df.loc[df["is_jump"], "MMSI"].unique()
    df = df[~df["MMSI"].isin(vessels_with_residual_jump)]

    df = df.drop(columns=["interping_duration", "interping_distance", "interping_speed", "is_jump"])

    # --- Drop rows with remaining NaNs ---
    df = df.dropna()

    return df


def resample_pings(df, interval=None):
    """
    Resample sequence of pings to fixed time interval for each vessel.
    Potential gaps are linearly interpolated.
    First value of each bin is used as the bin value
    (consistent with left-closed bin index).
    """
    if interval is None:
        interval = f"{settings.resample_interval_min}min"

    df = df.copy()
    df["BaseDateTime"] = pd.to_datetime(df["BaseDateTime"])
    df = df.sort_values(["MMSI", "BaseDateTime"]).set_index("BaseDateTime")

    # First ping of each bin assigns the entire bin value
    df_resampled = df.groupby("MMSI").resample(interval).first()
    # MMSI is in the multiindex after groupby; drop redundant column
    df_resampled = df_resampled.drop(columns=["MMSI"])

    linear_cols = ["LAT", "LON", "SOG", "Length", "Width", "Draft"]
    hold_cols = ["Heading", "Status"]

    g = df_resampled.groupby("MMSI")
    for col in linear_cols:
        df_resampled[col] = g[col].transform(lambda s: s.interpolate("linear"))
    for col in hold_cols:
        df_resampled[col] = g[col].transform("ffill")

    return df_resampled.reset_index()


def vessel_train_test_split(df, test_size=0.15, val_size=0.15, random_state=None):
    """Split dataset by vessel groups into train/val/test sets.

    Ensures no vessel appears in multiple sets to prevent data leakage.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with MMSI column
    test_size : float, default=0.15
        Proportion of vessels (not rows) to put in test set
    val_size : float, default=0.15
        Proportion of vessels (of total) to put in validation set
    random_state : int or None, optional
        Random seed for reproducibility. If None, uses settings.random_seed.

    Returns:
    --------
    tuple
        (df_train, df_val, df_test, groups_train, groups_val, groups_test)
    """
    if random_state is None:
        random_state = settings.random_seed

    if "MMSI" not in df.columns:
        raise ValueError("DataFrame must contain 'MMSI' column")

    groups = df["MMSI"]

    # First split: separate test from train+val
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    for train_idx, test_idx in gss.split(df, y=None, groups=groups):
        df_train_val, df_test = df.iloc[train_idx], df.iloc[test_idx]
        groups_train_val = groups.iloc[train_idx]
        groups_test = groups.iloc[test_idx]

    # Second split: separate train from val (on train+val subset)
    gss = GroupShuffleSplit(
        n_splits=1, test_size=val_size / (1 - test_size), random_state=random_state
    )
    for train_idx, val_idx in gss.split(df_train_val, y=None, groups=groups_train_val):
        df_train, df_val = df_train_val.iloc[train_idx], df_train_val.iloc[val_idx]
        groups_train = groups_train_val.iloc[train_idx]
        groups_val = groups_train_val.iloc[val_idx]

    mmsi_train = set(df_train["MMSI"].unique())
    mmsi_val = set(df_val["MMSI"].unique())
    mmsi_test = set(df_test["MMSI"].unique())
    overlap_train_val = mmsi_train & mmsi_val
    overlap_train_test = mmsi_train & mmsi_test
    overlap_val_test = mmsi_val & mmsi_test

    if overlap_train_val or overlap_train_test or overlap_val_test:
        raise ValueError(
            "MMSI leakage between splits: "
            f"train∩val={len(overlap_train_val)}, "
            f"train∩test={len(overlap_train_test)}, "
            f"val∩test={len(overlap_val_test)}"
        )

    return df_train, df_val, df_test, groups_train, groups_val, groups_test
