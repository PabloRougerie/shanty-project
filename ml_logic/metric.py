import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import make_scorer


def haversine_distance(LAT_true, LON_true, LAT_pred, LON_pred):
    """
    Calculate great-circle distance between two points using Haversine formula.

    Parameters:
    LAT_true, LON_true : array. True latitude and longitude coordinates in degrees
    LAT_pred, LON_pred : array. Predicted latitude and longitude coordinates in degrees

    Returns:
    --------
    float or array-like
        Great-circle distance in kilometers
    """
    earth_radius = 6371  # Earth mean radius in kilometers

    # Convert degrees to radians
    LAT_true_rad = np.radians(LAT_true)
    LON_true_rad = np.radians(LON_true)
    LAT_pred_rad = np.radians(LAT_pred)
    LON_pred_rad = np.radians(LON_pred)

    # Calculate differences
    d_LAT = LAT_pred_rad - LAT_true_rad
    d_LON = LON_pred_rad - LON_true_rad

    # Haversine formula
    a = (np.sin(d_LAT / 2.0)**2 +
         np.cos(LAT_true_rad) * np.cos(LAT_pred_rad) * np.sin(d_LON / 2.0)**2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))  # Angular distance in radians
    d = c * earth_radius  # Convert to kilometers

    return d


def haversine_mae(y_true, y_pred):
    """
    Calculate MAE using Haversine distance between positions.
    Evaluates prediction quality in km rather than degrees.

    Parameters:

    y_true : array, shape (n_samples, 2)
    True positions where column 0 = LAT, column 1 = LON (in degrees)
    y_pred : array, shape (n_samples, 2)
    Predicted positions where column 0 = LAT, column 1 = LON (in degrees)

    Returns:
    float: Mean Absolute Error in kilometers
    """
    # Make sure we're converting to numpy
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Calculate haversine distance for each prediction
    distances = haversine_distance(
        y_true[:, 0], y_true[:, 1],
        y_pred[:, 0], y_pred[:, 1])

    mae = np.mean(np.abs(distances))

    return mae

# Create  scorer
haversine_scorer = make_scorer(haversine_mae, greater_is_better= False)


def position_extrapolation(df, time_horizon):
    """
    Naive baseline: extrapolate position at time t + time_horizon based on
    linear displacement between time t - time_horizon and t.

    Assumes constant velocity: future_displacement = past_displacement
    Mathematically: position(t + horizon) = position(t) + [position(t) - position(t - horizon)]

    Parameters:

    df : DataFrame
        Must contain columns: LAT, LON, LAT_lag_{time_horizon}min, LON_lag_{time_horizon}min
    time_horizon : int
        Prediction horizon in minutes (must match the lag column names)

    Returns:
    --------
    LAT_pred, LON_pred : Series
        Predicted latitude and longitude at time t + time_horizon
    """
    # Calculate displacement over the last time_horizon minutes
    dLAT = df["LAT"] - df[f"LAT_lag_{time_horizon}min"]
    dLON = df["LON"] - df[f"LON_lag_{time_horizon}min"]

    # Extrapolate: assume same displacement for next time_horizon minutes
    LAT_pred = df["LAT"] + dLAT
    LON_pred = df["LON"] + dLON

    return LAT_pred, LON_pred


# creeer scorer haversine tensor flow
class HaversineMAE(tf.keras.metrics.Metric):
    """Metric for TF, bc cannot use the sklearn version.

    Calculates great-circle distance between true and predicted (LAT, LON) positions,
    then returns the mean distance across all samples. Useful for geospatial predictions.
    """

    #constructor for instantiation
    def __init__(self, name= "HaversineMAE", **kwargs): #pass all possible arguments from parent class

        super().__init__(name= name, **kwargs) #inherit the constructor from parent class

        #sum of individual MAE (by batch?)
        self.total = self.add_weight(name= "total", initializer= "zeros")
        #count of number of elements (or batch?)
        self.count = self.add_weight(name="count", initializer= "zeros")

    #called after each batch, calculate haversine MAE for this batch and store
    def update_state(self, y_true, y_pred, sample_weight= None):

        # Extract LAT and LON from y_true and y_pred (shape: (batch_size, 2))
        # Column 0 = LAT, Column 1 = LON (in degrees, not scaled)
        lat1 = y_true[:, 0]  # True latitude in degrees
        lon1 = y_true[:, 1]  # True longitude in degrees
        lat2 = y_pred[:, 0]  # Predicted latitude in degrees
        lon2 = y_pred[:, 1]  # Predicted longitude in degrees

        # Convert degrees to radians
        deg2rad = tf.constant(np.pi / 180.0, dtype=tf.float32)
        lat1_rad = lat1 * deg2rad
        lon1_rad = lon1 * deg2rad
        lat2_rad = lat2 * deg2rad
        lon2_rad = lon2 * deg2rad

        # Calculate differences in radians
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        # Haversine formula
        a = tf.sin(dlat/2)**2 + tf.cos(lat1_rad) * tf.cos(lat2_rad) * tf.sin(dlon/2)**2
        a = tf.clip_by_value(a, 0.0, 1.0) #safety for edge cases
        c = 2 * tf.asin(tf.sqrt(a))
        distance = 6371 * c  # Distance in kilometers

        #sum all distances from the current batch and do a cumulative sum with the previous total
        self.total.assign_add(tf.reduce_sum(distance)) #tf equivalent of +=
        #
        self.count.assign_add(tf.cast(tf.size(distance), tf.float32)) #nb of element and cast it to a float32 format

    #return MAE, can be called after each batch for display or each epoch for looging
    def result(self):
        return self.total / self.count #average

    #called at beginning of each epoch. reset mae
    def reset_state(self):
        self.total.assign(0.)
        self.count.assign(0.)
