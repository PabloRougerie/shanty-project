import pandas as pd
import numpy as np
import gc
import requests
from datetime import datetime, timedelta
from pathlib import Path
import io
import zipfile
import os


from sklearn.model_selection import GroupShuffleSplit



def download_source_files(start_date: str, end_date, lon_west, lon_east, lat_north, lat_south, output_path= "../data/"):
    """
    https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2024/AIS_2024_12_31.zip
    """

    #convert date into date time
    start_date = pd.to_datetime(start_date, format="%Y-%m-%d")
    end_date = pd.to_datetime(end_date, format="%Y-%m-%d")
    temp_dir_path = Path(output_path)/"temp"
    temp_dir_path.mkdir(parents= True, exist_ok= True)

    current_date = start_date

    while current_date <= end_date:

        year = current_date.year
        month = current_date.month
        day = current_date.day

        # 1. FILE DOWNLOAD WITHOUT SAVING CSV (we'll filtered first)
        # construct URL to fetch files on NOAA according to their naming convention
        url = f"https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{year}/AIS_{year}_{month:02d}_{day:02d}.zip"

        #download zipfiles. stream= True bc larger files
        print(f"Requestion zip files for date {current_date}")
        response= requests.get(url, stream= True)
        response.raise_for_status() #raise exception if http requests failes
        zip_data = io.BytesIO(response.content) #store content of request in RAM

        # READ CSV FROM ZIP

        with zipfile.ZipFile(zip_data) as zip:
            filename = zip.namelist()[0] #get csv file name from zip files list

            #open the file from within the zipfile
            with zip.open(filename) as csv_file:
                df = pd.read_csv(csv_file)

        #FILTER GEOGRAPHIC AREA OF INTEREST
        #doing it right away to minimize file size to be saved to disk.
        #Lon are negative west of greenwhich
        print(f"Restricting to AOI")
        mask = (df["LON"] >= lon_west) & (df["LON"] <= lon_east) & (df["LAT"] >= lat_south) & (df["LAT"] <= lat_north)
        df_bound = df.loc[mask,:]

        #FILTER BY VESSEL TYPE
        print(f"Selecting cargo ships...")
        df_bound = df_bound.loc[ (df_bound["VesselType"] >= 70) & (df_bound["VesselType"] < 90),:]

        #SELECT RELEVANT COLUMNS
        print(f"Filtering features...")
        df_filtered = df_bound.drop(columns= ["VesselName", "IMO", "CallSign",
                                              "VesselType",
                                              "Cargo",
                                              "TransceiverClass"])
        #SAVE TEMP PARQUET FILE
        #save daily log as parquet with 2 digit formation for months and days
        temp_parquet = temp_dir_path / f"temp_AIS_{year}_{month:02d}_{day:02d}.parquet"
        df_filtered.to_parquet(temp_parquet, index= False)

        del df, df_bound, df_filtered, zip_data
        gc.collect()
        current_date += timedelta(days= 1)

    # find all parquet files in temp dir (sorted by filename, hence by date)
    temp_parquet_files = sorted(temp_dir_path.glob("*.parquet"))

    #security check if there is no file
    if not temp_parquet_files:
        print("No parquet files found in temp dir")
        return

    print(f"Merging {len(temp_parquet_files)} parquet files into a dataframe")
    #read all daily parquet files
    dataframes = [pd.read_parquet(file) for file in temp_parquet_files]

    #merge
    #concatenate vertically
    df_merged = pd.concat(dataframes, ignore_index=True, axis= 0)
    raw_dir_path = Path(output_path) / "raw"
    raw_dir_path.mkdir(parents=True, exist_ok=True)
    parquet_merged_filename = raw_dir_path / "AIS_merged.parquet"
    df_merged.to_parquet(parquet_merged_filename, index= False)

    #clean tem files
    for temp_file in temp_parquet_files:
        #delete temp files
        temp_file.unlink()
    print(f"Deleted {len(temp_parquet_files)} temporary files.")

    #clean memory
    del dataframes, df_merged
    gc.collect()


def get_raw_data(target_path, source_path= "../data/raw/AIS_.csv"):
    """ get the raw csv file, select relevant columns,
    restrict to cargo and tanker tracks,
    and convert to .parquet"""

    df_raw = pd.read_csv(source_path)
    print(df_raw.columns)
    #identify features worth keeping
    col_to_keep = ["MMSI", "BaseDateTime", "LAT", "LON", "SOG", "COG", "Heading", "VesselType", "Status", "Length", "Width", "Draft"]
    df_reduced = df_raw[col_to_keep]
    #select only cargo and tankers
    df_reduced = df_reduced.loc[ (df_reduced["VesselType"] >= 70) & (df_reduced["VesselType"] < 90),:]
    df_reduced.to_parquet()
    del df_raw
    gc.collect()


def clean_data(df):
    """Remove duplicates and handle initial missing values"""

    df = df.drop_duplicates()
    #df = df.drop(columns=["VesselType"])

    # Replace 0 values with NaN for vessel dimensions (0 = missing data, not real dimensions)
    dimension_cols = ["Length", "Width", "Draft"]
    df[dimension_cols] = df[dimension_cols].replace(0, np.nan)

    # Impute missing dimensions with median
    na_values = {"Length": df["Length"].median(),
             "Width": df["Width"].median(),
             "Draft": df["Draft"].median()}

    df = df.fillna(value=na_values)
    df = df.dropna(subset=["Status"])

    return df

def resample_pings(df, interval='5min'):
    """
    Resample sequence of pings to fixed time interval for each vessel.
    Potential gaps are linearly interpolated.
    Last value of each bin is used to determine the bin value.
    """

    df["BaseDateTime"] = pd.to_datetime(df["BaseDateTime"])
    df = df.sort_values(by=["MMSI", "BaseDateTime"], ascending=True)
    df = df.set_index("BaseDateTime")

    # Use the interval parameter
    df_resampled = df.groupby("MMSI").resample(interval).last()
    df_resampled = df_resampled.interpolate("linear")
    df_resampled = df_resampled.drop(columns=["MMSI"])
    df = df_resampled.reset_index()

    return df


def create_target(df: pd.DataFrame, horizon: int):
    """Create target columns by forward-shifting LAT and LON for each vessel.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns MMSI, BaseDateTime, LAT, LON
    horizon : int
        Number of time steps to shift forward (prediction horizon)

    Returns:
    --------
    pd.DataFrame
        DataFrame with added target_LAT and target_LON columns, rows with NaN targets removed
    """

    required_cols = ["MMSI", "BaseDateTime", "LAT", "LON"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    #sort by vessels and time
    df = df.sort_values(by=["MMSI", "BaseDateTime"], ascending=True)

    #create target columns by shifting forward the LAT and LON
    df['target_LAT'] = df.groupby('MMSI')['LAT'].shift(-horizon)
    df['target_LON'] = df.groupby('MMSI')['LON'].shift(-horizon)

    #clean rows with Nan
    df = df.dropna(axis= 0)

    return df

def vessel_train_test_split(df, test_size= 0.2, val_size= 0.15, random_state = 273):
    """Split dataset by vessel groups into train/val/test sets.

    Ensures no vessel appears in multiple sets to prevent data leakage.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with MMSI column
    test_size : float, default=0.2
        Proportion of vessels (not rows) to put in test set
    val_size : float, default=0.15
        Proportion of vessels from train+val to put in validation set
    random_state : int, optional
        Random seed for reproducibility

    Returns:
    --------
    tuple
        (df_train, df_val, df_test, groups_train, groups_val, groups_test)
    """
    if "MMSI" not in df.columns:
        raise ValueError("DataFrame must contain 'MMSI' column")

    groups = df["MMSI"]

    # First split: separate test from train+val
    gss = GroupShuffleSplit(n_splits= 1, test_size= test_size, random_state= random_state)
    for train_idx, test_idx in gss.split(df, y= None, groups= groups):
        df_train_val, df_test = df.iloc[train_idx], df.iloc[test_idx]
        groups_train_val = groups.iloc[train_idx]
        groups_test = groups.iloc[test_idx]

    # Second split: separate train from val (on train+val subset)
    gss = GroupShuffleSplit(n_splits= 1, test_size= val_size, random_state= random_state)
    for train_idx, val_idx in gss.split(df_train_val, y= None, groups= groups_train_val):
        df_train, df_val = df_train_val.iloc[train_idx], df_train_val.iloc[val_idx]
        groups_train = groups_train_val.iloc[train_idx]
        groups_val = groups_train_val.iloc[val_idx]

    return df_train, df_val, df_test, groups_train, groups_val, groups_test


def get_eligible_vessels(df, lookback, horizon, min_nb_seq= 200):
    """Filter vessels that have enough time steps to create sequences.

    An eligible sequence requires (lookback + horizon) time steps.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with MMSI column
    lookback : int
        Length of input sequence
    horizon : int
        Prediction horizon (already accounted for in create_target)
    min_nb_seq : int, default=200
        Minimum number of sequences required per vessel

    Returns:
    --------
    list
        List of MMSI values for eligible vessels
    """
    if "MMSI" not in df.columns:
        raise ValueError("DataFrame must contain 'MMSI' column")

    #pd.Series of vessel-wise full track duration
    track_duration = df["MMSI"].value_counts()

    #pd.Series of vessel-wise number of eligible time sequences
    nb_of_seq = np.maximum(0, track_duration - (lookback + horizon))

    vessels_list = nb_of_seq[nb_of_seq > min_nb_seq].index.to_list()

    if len(vessels_list) == 0:
        print(f"Warning: No vessels found with at least {min_nb_seq} sequences")

    return vessels_list



def create_sliding_windows(df, lookback, vessel_list):
    """Create sliding time windows from vessel tracks for LSTM input.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with MMSI, BaseDateTime, target_LAT, target_LON, and feature columns
    lookback : int
        Length of input sequence (number of time steps)
    vessel_list : list
        List of MMSI values to process

    Returns:
    --------
    tuple of np.ndarray
        X: shape (n_sequences, lookback, n_features)
        y: shape (n_sequences, 2) for [LAT, LON]
    """
    required_cols = ["MMSI", "BaseDateTime", "target_LAT", "target_LON"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if not vessel_list:
        raise ValueError("vessel_list cannot be empty")

    #select df with only eligible vessels
    df_restricted = df.loc[df["MMSI"].isin(vessel_list),:]

    #make sure we're sorted by vessels and dates: from older to more recent date
    df_sorted = df_restricted.sort_values(by=["MMSI", "BaseDateTime"], ascending=True)

    # Pre-compute columns to keep for X (exclude target and MMSI)
    cols_to_drop = required_cols
    feature_cols = [col for col in df_sorted.columns if col not in cols_to_drop]



    X_time_sequences = [] #we'll stock time sequences there
    y_time_sequences = []

    total_vessels = len(vessel_list)
    print(f"Creating sliding windows for {total_vessels} vessels...")

    # Use groupby once instead of filtering in loop (much faster)
    vessel_idx = 0
    for vessel, vessel_group in df_sorted.groupby("MMSI"):


        vessel_idx += 1
        vessel_track = vessel_group[feature_cols].values  # Convert to numpy array once
        vessel_targets = vessel_group[["target_LAT", "target_LON"]].values
        nb_sequences_vessel = len(vessel_track) - lookback  # lookback past + 1 present = lookback+1 total

        if vessel_idx % 50 == 0 or vessel_idx == total_vessels:
            print(f"  Processing vessel {vessel_idx}/{total_vessels} (MMSI: {vessel}) - {nb_sequences_vessel} sequences")

        # Vectorized sliding window creation using numpy slicing
        # Lookback includes past time steps + current time step
        for i in range(nb_sequences_vessel):
            X_seq = vessel_track[i: i + lookback + 1]  # lookback past steps + current time step
            y_seq = vessel_targets[i + lookback]  # Target from current time step (which contains future position)
            X_time_sequences.append(X_seq)
            y_time_sequences.append(y_seq)

    # Convert to numpy arrays: X shape (n_sequences, lookback, n_features), y shape (n_sequences, 2)
    print(f"Converting {len(X_time_sequences)} sequences to numpy arrays...")
    X_array = np.array(X_time_sequences)
    y_array = np.array(y_time_sequences)

    if len(X_array) == 0:
        raise ValueError("No sequences created. Check lookback and vessel_list.")

    return X_array, y_array



def create_LSTM_sets(df: pd.DataFrame,
                     lookback: int,
                     horizon: int,
                     test_size= 0.2,
                     val_size= 0.15,
                     random_state= 273,
                     min_nb_seq= 200):
    """Prepare train/val/test sets for LSTM from AIS vessel tracking data.

    Pipeline: create targets → split by vessel → filter eligible vessels → create sequences.

    Parameters:
    -----------
    df : pd.DataFrame
        Preprocessed DataFrame with MMSI, BaseDateTime, LAT, LON, and features
    lookback : int
        Length of input sequence (time steps)
    horizon : int
        Prediction horizon (time steps ahead)
    test_size : float, default=0.2
        Proportion of vessels for test set (0-1)
    val_size : float, default=0.15
        Proportion of vessels from train+val for validation set (0-1)
    random_state : int, default=273
        Random seed for train/val/test split
    min_nb_seq : int, default=200
        Minimum sequences per vessel to be included

    Returns:
    --------
    tuple of np.ndarray
        X_train, y_train, X_val, y_val, X_test, y_test
        Shapes: X (n_sequences, lookback, n_features), y (n_sequences, 2)
    """


    #create the target
    df_with_target = create_target(df, horizon= horizon)
    print(f"Created targets: {len(df_with_target)} rows remaining after shift(-{horizon})")

    #split into train, val, and test sets
    df_train, df_val, df_test, groups_train, groups_val, groups_test = vessel_train_test_split(df= df_with_target,
                                                                           test_size= test_size,
                                                                           val_size= val_size,
                                                                            random_state= random_state)
    print(f"Train: {len(df_train)} rows, {df_train['MMSI'].nunique()} vessels | "
          f"Val: {len(df_val)} rows, {df_val['MMSI'].nunique()} vessels | "
          f"Test: {len(df_test)} rows, {df_test['MMSI'].nunique()} vessels")

    #get lists of eligible vessels for future use
    vessel_train = get_eligible_vessels(df_train, lookback= lookback,
                                        horizon= horizon, min_nb_seq= min_nb_seq)
    vessel_val = get_eligible_vessels(df_val, lookback= lookback,
                                        horizon= horizon, min_nb_seq= min_nb_seq)
    vessel_test = get_eligible_vessels(df_test, lookback= lookback,
                                        horizon= horizon, min_nb_seq= min_nb_seq)

    print(f"Eligible vessels: {len(vessel_train)} train, {len(vessel_val)} val, {len(vessel_test)} test")

    print("\n=== Creating TRAIN sequences ===")
    X_train_seq, y_train_seq = create_sliding_windows(df_train, lookback= lookback, vessel_list= vessel_train)
    print(f"✓ Train sequences created: {X_train_seq.shape[0]} sequences\n")

    print("=== Creating VALIDATION sequences ===")
    X_val_seq, y_val_seq = create_sliding_windows(df_val, lookback= lookback, vessel_list= vessel_val)
    print(f"✓ Val sequences created: {X_val_seq.shape[0]} sequences\n")

    print("=== Creating TEST sequences ===")
    X_test_seq, y_test_seq = create_sliding_windows(df_test, lookback= lookback, vessel_list= vessel_test)
    print(f"✓ Test sequences created: {X_test_seq.shape[0]} sequences\n")

    print(f"Final shapes - X_train: {X_train_seq.shape}, y_train: {y_train_seq.shape} | "
          f"X_val: {X_val_seq.shape}, y_val: {y_val_seq.shape} | "
          f"X_test: {X_test_seq.shape}, y_test: {y_test_seq.shape}")

    return X_train_seq, y_train_seq, X_val_seq, y_val_seq, X_test_seq, y_test_seq
