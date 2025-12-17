import pandas as pd
import numpy as np
from ml_logic.metric import haversine_distance

def create_time_series_features(df, target_horizon=30, rolling=False, advanced_features=False):
    """
    Create lagged features and target proportional to prediction horizon.
    Optionally add rolling statistics and advanced engineered features.

    Lag times are 1/6th, 1/2, and 1/1 of the target time horizon.

    Parameters:
    -----------
    df : DataFrame
        Must be resampled at 5-min intervals, sorted by MMSI and BaseDateTime
    target_horizon : int
        Prediction horizon in minutes (30, 60, 360,..)
    rolling : bool, default False
        If True, add rolling statistics (mean, std) over target_horizon window
    advanced_features : bool, default False
        If True, add advanced engineered features (rate of change, ratios, meandering)

    Returns:
    --------
    DataFrame with lagged features, rolling features (optional), advanced features (optional), and targets
    """
    df_lag = df.copy()

    # Count number of 5min steps in the time horizon
    nb_steps = target_horizon // 5

    #determine the number of step in the lag windows
    lag_1 = max(1,nb_steps//6) #max to ensure there's at least one step in the window
    lag_2 = max(2, nb_steps // 2)
    lag_3 = nb_steps #full duration of the target horizon (for symmetry)

    print(f"Target prediction horizon: {target_horizon} min. Number of steps: {nb_steps}")
    print(f"Defining lag windows of {lag_1*5}min, {lag_2*5}min, {lag_3*5}min")

    # Creating lag features
    feats = ["LAT", "LON", "SOG", "COG"]
    for feat in feats:
        df_lag[f"{feat}_lag_{lag_1*5}min"] = df_lag.groupby("MMSI")[feat].shift(lag_1)
        df_lag[f"{feat}_lag_{lag_2*5}min"] = df_lag.groupby("MMSI")[feat].shift(lag_2)
        df_lag[f"{feat}_lag_{lag_3*5}min"] = df_lag.groupby("MMSI")[feat].shift(lag_3)

    # Create rolling features (BEFORE dropna/reset_index to preserve index alignment)
    if rolling:
        print(f"Adding rolling features over {target_horizon}min window")
        df_lag["SOG_rolling_mean"] = df_lag.groupby("MMSI")["SOG"].rolling(window=nb_steps, min_periods=1
        ).mean().reset_index(level=0, drop=True)

        df_lag["SOG_rolling_std"] = df_lag.groupby("MMSI")["SOG"].rolling(window=nb_steps, min_periods=1
        ).std().reset_index(level=0, drop=True)

        df_lag["COG_rolling_std"] = df_lag.groupby("MMSI")["COG"].rolling(window=nb_steps, min_periods=1
        ).std().reset_index(level=0, drop=True)

    # Create advanced features
    if advanced_features:
        print(f"Adding advanced engineered features")

         #. Rate of change (using shortest lag window)
        df_lag["SOG_change"] = df_lag["SOG"] - df_lag[f"SOG_lag_{lag_1*5}min"]

        # 2. Geometric ratios (vessel maneuverability and stability)
        df_lag["Length_Width_ratio"] = df_lag["Length"] / df_lag["Width"]
        df_lag["Draft_Width_ratio"] = df_lag["Draft"] / df_lag["Width"]
        df_lag["vessel_volume"] = df_lag["Draft"] * df_lag["Length"] * df_lag["Width"]

        # 3. Meandering index (trajectory complexity using longest lag window)
         #Distance travelled (theoretical from SOG)
        # SOG in knots * time in hours = distance in nautical miles, then convert to km
        distance_travelled_nm = df_lag["SOG"] * (lag_3 * 5 / 60)  # nautical miles
        distance_travelled_km = distance_travelled_nm * 1.852  # convert to km (1 NM = 1.852 km)

        # Direct distance (haversine between current and lag_3 position) in km
        direct_distance = haversine_distance(
            df_lag["LAT"], df_lag["LON"],
            df_lag[f"LAT_lag_{lag_3*5}min"], df_lag[f"LON_lag_{lag_3*5}min"])

        # Meandering index: direct / travelled (1 = straight line, <1 = sinuous trajectory)
        # Use max(distance_travelled_km, 0.1) to avoid division by zero (stationary vessels)
        # Clip between 0 and 1
        df_lag["meandering_index"] = np.clip(direct_distance / np.maximum(distance_travelled_km, 0.1),0, 1)

    # Create target
    df_lag['target_LAT'] = df_lag.groupby('MMSI')['LAT'].shift(-nb_steps)
    df_lag['target_LON'] = df_lag.groupby('MMSI')['LON'].shift(-nb_steps)

    # Clean the df
    df_lag = df_lag.dropna().reset_index(drop=True)

    return df_lag
