import pandas as pd
import numpy as np
import gc

def get_raw_data(url= "../data/raw/AIS_.csv"):
    """ get the raw csv file, select relevant columns,
    restrict to cargo and tanker tracks,
    and convert to .parquet"""

    df_raw = pd.read_csv(url)
    #identify features worth keeping
    col_to_keep = ["MMSI", "BaseDateTime", "LAT", "LON", "SOG", "COG", "Heading", "VesselType", "Status", "Length", "Width", "Draft"]
    df_reduced = df_raw[col_to_keep]
    #select only cargo and tankers
    df_reduced = df_reduced.loc[ (df_reduced["VesselType"] >= 70) & (df_reduced["VesselType"] < 90),:]
    df_reduced.to_parquet("../data/processed/ais_filtered.parquet")
    del df_raw
    gc.collect()



def clean_data(df):
    """Remove duplicates and handle initial missing values"""

    df = df.drop_duplicates()
    df = df.drop(columns=["VesselType"])

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
