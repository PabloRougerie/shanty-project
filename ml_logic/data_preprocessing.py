import pandas as pd
import numpy as np
import gc
import requests
from datetime import datetime, timedelta
from pathlib import Path
import io
import zipfile
import os



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


def create_time_window(X: pd.DataFrame, y: pd.Series, size= 50):

    """
    Splits the input DataFrame X and Series y into overlapping time windows of a given size.
    The split has to make sure that time sequences of separate vessels aren't mixed

    Parameters:
    -----------
    X : pd.DataFrame
        Feature data to be windowed.
    y : pd.Series
        Target variable to be windowed.
    size : int
        Length of each time window in time steps (5min by default).

    Returns:
    --------
    tuple of np.ndarray
        Two arrays:
        - X_timeframes: shape (num_windows, size, num_features)
        - y_timeframes: shape (num_windows, size)
    """

    # le dataset a déjà le double target obtenu par shift

    # filtrer les bateaux
    # generer la target avec le shift

    #




    #get a list of vessels that have been tracked for at least size time step
    vessel_list = X.groupby('MMSI').filter(lambda x: len(x) > size)
    print(f"Number of vessels kept: {len(vessel_list)}")

    # OPTIONAL : INCLUDE A QUALITY CHECK IN CASE OF THERE ARENT ENOUGH VESSEL LEFT

    #storage for the time sequences: overlapping time sequences of all vessels are appended in same list
    X_timeframes = []
    y_timeframes = []
    # For each vessel, create overlapping time sequence
    for vessel in vessel_list:

        vessel_X = X[""]









        #create sliding time windows by slicing dataframes in index number
        for i in range(total_duration - size):

            X_timeslice = X.iloc[i:i+size, :]
            y_timeslice = y.iloc[i:i+size]

            X_timeframes.append(X_timeslice)
            y_timeframes.append(y_timeslice)



        print(f"created {len(X_timeframes)} time windows")
        print(f" output is of shape {np.array(X_timeframes).shape}")

        return np.array(X_timeframes), np.array(y_timeframes)
