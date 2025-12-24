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
from sklearn.preprocessing import RobustScaler



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

    # Create descriptive filename with bounding box coordinates and date range
    # Format coordinates: lon{west}to{east}_lat{south}to{north}
    # Format dates: {start_date}to{end_date} (YYYYMMDD)
    coords_str = f"lon{lon_west:.1f}to{lon_east:.1f}_lat{lat_south:.1f}to{lat_north:.1f}"
    dates_str = f"{start_date.strftime('%Y%m%d')}to{end_date.strftime('%Y%m%d')}"
    parquet_merged_filename = raw_dir_path / f"AIS_merged_{coords_str}_{dates_str}.parquet"

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

    # ====== SAFETY CHECK ======
    # Verify required columns exist in DataFrame
    required_cols = ["MMSI", "BaseDateTime", "target_LAT", "target_LON"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if not vessel_list:
        raise ValueError("vessel_list cannot be empty")

    # ====== PREPARE DATASET ======
    # Filter DataFrame to only eligible vessels (faster than filtering in loop)
    df_restricted = df.loc[df["MMSI"].isin(vessel_list), :]

    # Sort by vessel and time: essential for correct sliding window creation
    # Must be chronological (oldest to newest) for each vessel
    df_sorted = df_restricted.sort_values(by=["MMSI", "BaseDateTime"], ascending=True)

    # Identify feature columns: exclude metadata (MMSI, BaseDateTime) and targets
    cols_to_drop = ["MMSI", "BaseDateTime", "target_LAT", "target_LON"]
    feature_cols = [col for col in df_sorted.columns if col not in cols_to_drop]

    total_vessels = len(vessel_list)
    print(f"Creating sliding windows for {total_vessels} vessels...")


    # ====== CREATE SLIDING WINDOWS ======
    vessel_idx = 0
    # Store pre-allocated arrays per vessel (will concatenate at the end)
    X_arrays = []
    y_arrays = []

    # Use groupby once: iterates over vessels without filtering DataFrame each time
    # Much faster than df.loc[df["MMSI"] == vessel] in a loop
    for vessel, vessel_group in df_sorted.groupby("MMSI"):

        vessel_idx += 1

        # Convert vessel data to numpy arrays once (faster than pandas operations in loop)
        # vessel_track: all feature observations for this vessel, shape (n_time_steps, n_features)
        vessel_track = vessel_group[feature_cols].values

        # vessel_targets: all target positions for this vessel, shape (n_time_steps, 2)
        # Each row contains [target_LAT, target_LON] for that time step
        vessel_targets = vessel_group[["target_LAT", "target_LON"]].values

        # Calculate number of sequences: need at least (lookback + 1) time steps per sequence
        # Formula: if vessel has N time steps, can create (N - lookback) sequences
        nb_sequences_vessel = len(vessel_track) - lookback

        if vessel_idx % 50 == 0 or vessel_idx == total_vessels:
            print(f"  Processing vessel {vessel_idx}/{total_vessels} (MMSI: {vessel}) - {nb_sequences_vessel} sequences")

        # Create sliding windows: each sequence = lookback past steps + current time step
        if nb_sequences_vessel > 0:
            # Pre-allocate arrays for this vessel (faster than appending to Python lists)
            # X_vessel shape: (nb_sequences, lookback+1, n_features)
            # y_vessel shape: (nb_sequences, 2) for [LAT, LON]
            n_features = vessel_track.shape[1]
            X_vessel = np.empty((nb_sequences_vessel, lookback + 1, n_features))
            y_vessel = np.empty((nb_sequences_vessel, 2))

            # Fill arrays by slicing vessel_track: each sequence overlaps with previous one
            for i in range(nb_sequences_vessel):
                # X: slice from index i to i+lookback+1 (includes current time step)
                # Shape: (lookback+1, n_features) → stored in X_vessel[i]
                X_vessel[i] = vessel_track[i: i + lookback + 1]

                # y: target is at index i+lookback (current time step, which contains future position)
                # Shape: (2,) for [LAT, LON] → stored in y_vessel[i]
                y_vessel[i] = vessel_targets[i + lookback]

            # Store vessel arrays in list (will concatenate all vessels at the end)
            X_arrays.append(X_vessel)
            y_arrays.append(y_vessel)

    # Concatenate all vessel arrays along axis=0 (stack vertically)
    # Faster than np.array(list) because arrays are already allocated
    print(f"Concatenating {len(X_arrays)} vessel arrays...")
    X_array = np.concatenate(X_arrays, axis=0)  # Final shape: (total_sequences, lookback+1, n_features)
    y_array = np.concatenate(y_arrays, axis=0)  # Final shape: (total_sequences, 2)

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
    print(f"group_train: {groups_train}")

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


def reshape_helper(X: np.ndarray, fit=False, scaler = RobustScaler()):
    """Helper function to scale 3D LSTM sequences.

    Reshapes 3D array to 2D for scaling, then reshapes back to preserve sequence structure.

    Parameters:
    -----------
    X : np.ndarray
        3D array of shape (n_sequences, seq_length, n_features)
    fit : bool, default=False
        If True, fit scaler on data. If False, only transform
    scaler: sklearn scaler (default, Robust Scaler)

    Returns:
    --------
    np.ndarray
        Scaled array with same shape as input
    """

    print(f"Scaling dataset of shape {X.shape}...")

    # Reshape to 2D: (n_sequences * seq_length, n_features) for scaling
    X_2D = X.reshape(-1, X.shape[-1])

    if fit:
        # Fit scaler on data (for training set)
        scaler.fit(X_2D)

    # Transform data
    X_2D_sc = scaler.transform(X_2D)

    # Reshape back to original 3D shape
    X_sc = X_2D_sc.reshape(X.shape)

    return X_sc



def scale_LSTM_data(X_train: np.ndarray,
                    X_val: np.ndarray = None,
                    X_test: np.ndarray = None):
    """Scale LSTM input sequences using RobustScaler.

    Scales training set (fit + transform) and optionally validation/test sets (transform only).
    Reshapes 3D arrays to 2D for scaling, then reshapes back to preserve sequence structure.

    Parameters:
    -----------
    X_train : np.ndarray
        Training sequences, shape (n_sequences, seq_length, n_features)
    X_val : np.ndarray, optional
        Validation sequences, shape (n_sequences, seq_length, n_features)
    X_test : np.ndarray, optional
        Test sequences, shape (n_sequences, seq_length, n_features)

    Returns:
    --------
    tuple
        (X_train_scaled, X_val_scaled, X_test_scaled)
        Returns None for sets not provided
    """
    # Initialize return values (None if sets not provided)
    X_val_scaled = None
    X_test_scaled = None

    # Scale training set (fit scaler on this) with RobustScaler (by default in reshape_helper)
    print(f"Scaling train set, shape {X_train.shape}")
    X_train_scaled = reshape_helper(X_train, fit=True)

    # Scale validation set if provided (transform only, using scaler fitted on train)
    if X_val is not None:
        X_val_scaled = reshape_helper(X_val, fit=False)
        print(f"Scaling val set, shape {X_val.shape}")

    # Scale test set if provided (transform only, using scaler fitted on train)
    if X_test is not None:
        X_test_scaled = reshape_helper(X_test, fit=False)
        print(f"Scaling test set, shape {X_test.shape}")

    return X_train_scaled, X_val_scaled, X_test_scaled
