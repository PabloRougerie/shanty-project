import pandas as pd
import numpy as np

def clean_data(df):
    """remove duplicate and handle initial missing values"""

    df = df.drop_duplicates()
    df = df.drop(columns="VesselType")
    na_values = {"Length": df["Length"].median(),
             "Width": df["Width"].median(),
             "Draft": df["Draft"].median()}

    df = df.fillna(value= na_values)
    df = df.dropna(subset=["Status"])

    return df

def resample_pings(df, interval='5min'):
    """resample sequence of pings to fixed time interval for each vessels
    potential gaps are linearly interpolated.
    Last value of each bin is used to determine the bin value """

    df["BaseDateTime"] = pd.to_datetime(df["BaseDateTime"])
    df = df.sort_values(by=["MMSI", "BaseDateTime"], ascending= True)
    df = df.set_index("BaseDateTime")
    df_resampled = df.groupby("MMSI").resample("5min").last()
    df_resampled = df_resampled.interpolate("linear")
    df_resampled = df_resampled.drop(columns= "MMSI")
    df = df_resampled.reset_index()

    return df
