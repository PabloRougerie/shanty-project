import pandas as pd
import numpy as np

def create_time_series_features(df, target_horizon= 30):
    """
    Create lagged features and target proportional to prediction horizon.
    lag time are 1/6th, 1/2 and 1/1 of the target time horizon)

    Parameters:
    -----------
    df : DataFrame
        Must be resampled at 5-min intervals, sorted by MMSI and BaseDateTime
    prediction_horizon_min : int
        Prediction horizon in minutes (30, 60, 720, etc.)

    Returns:
    --------
    DataFrame with lagged features and targets
    """
    df_lag = df.copy()
    #count number of 5min steps in the time horizon
    nb_steps = target_horizon // 5

    #determine the number of step in the lag windows
    lag_1 = max(1,nb_steps//6) #max to ensure there's at least one step in the window
    lag_2 = max(2, nb_steps // 2)
    lag_3 = max(3,nb_steps) #full duration of the target horizon (for symmetry)

    print(f"Target prediction horizon: {target_horizon}. Number of steps: {nb_steps}")
    print(f"Defining lag windows of {lag_1 *5}min, {lag_2*5}min, {lag_3*5}min")

    #creating lag features
    feats = ["LAT", "LON", "SOG", "COG"]
    for feat in feats :
        df_lag[f"{feat}_lag_{lag_1*5}min"] = df_lag[feat].groupby("MMSI")[feat].shift(lag_1)
        df_lag[f"{feat}_lag_{lag_2*5}min"] = df_lag[feat].groupby("MMSI")[feat].shift(lag_2)
        df_lag[f"{feat}_lag_{lag_3*5}min"] = df_lag[feat].groupby("MMSI")[feat].shift(lag_3)

    #create target
    df_lag['target_LAT'] = df_lag.groupby('MMSI')['LAT'].shift(-nb_steps)
    df_lag['target_LON'] = df_lag.groupby('MMSI')['LON'].shift(-nb_steps)

    #clean the df
    df_lag = df_lag.dropna().reset_index(drop= True)

    return df_lag
