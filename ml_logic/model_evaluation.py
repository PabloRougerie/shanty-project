import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.multioutput import MultiOutputRegressor

from ml_logic.feature_engineering import create_time_series_features
from ml_logic.metric import position_extrapolation, haversine_mae

import tensorflow as tf

# ML: compare score between models and baseline for different prediction time horizon
def evaluate_horizon(df, time_horizon, estimators, test_size=0.2, random_state=None, rolling=False, advanced_features=False):
    """
    Evaluate baseline and ML models on test set for a given prediction horizon.

    Parameters:
    -----------
    df : DataFrame
        Preprocessed and resampled AIS data
    time_horizon : int
        Prediction horizon in minutes (30, 60, 360, 720, 1440, etc.)
    estimators : dict
        Dictionary of {'model_name': sklearn_estimator} to evaluate
    test_size : float, default=0.2
        Proportion of data for test set
    random_state : int or None, default=None
        Random state for reproducibility. If None, uses random split.
    rolling : bool, default=False
        If True, add rolling statistics features
    advanced_features : bool, default=False
        If True, add advanced engineered features (rate of change, ratios, meandering)

    Returns:
    --------
    nb_pings : int
        Total number of pings (samples) after feature engineering
    nb_vessels : int
        Total number of unique vessels in the dataset
    mae_scores : dict
        Dictionary with MAE scores for each model: {'Baseline': mae, 'ModelName': mae, ...}
    """
    # Create the df with the right lag windows
    df_lag = create_time_series_features(df, target_horizon=time_horizon, rolling=rolling, advanced_features=advanced_features)
    nb_pings = len(df_lag)
    nb_vessels = df_lag["MMSI"].nunique()

    # Separate X, y, groups
    X = df_lag.drop(columns=["MMSI", "BaseDateTime", "target_LAT", "target_LON"])
    y = df_lag[["target_LAT", "target_LON"]]
    groups = df_lag["MMSI"]

    # Train/test split respecting grouping by vessel
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)

    # Obtain the indices from the generator
    for train_idx, test_idx in gss.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Dictionary to store results
    mae_scores = {}

    # 1. BASELINE SCORE on the test set
    LAT_pred_baseline, LON_pred_baseline = position_extrapolation(X_test, time_horizon)
    y_pred_baseline = np.column_stack([LAT_pred_baseline, LON_pred_baseline])

    mae_baseline = haversine_mae(y_true=y_test.values, y_pred=y_pred_baseline)
    mae_scores["Baseline"] = mae_baseline

    # 2. ML MODELS SCORES on the test set
    for name, estimator in estimators.items():
        print(f"Testing model {name}")
        model = MultiOutputRegressor(estimator=estimator)
        print("fitting...")
        model.fit(X_train, y_train)
        print("predicting...")
        y_pred_model = model.predict(X_test)

        mae_model = haversine_mae(y_true=y_test.values, y_pred=y_pred_model)
        mae_scores[name] = mae_model

    return nb_pings, nb_vessels,mae_scores



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

        #target in shape (n, 2) where 0 is LAT, 1 is LON
        lat1 = y_true[:, 0]* 0.0174533 #conversion degree to radian (180 = pi rad)
        lon1 = y_true[:,1]* 0.0174533
        lat2 = y_pred[:,0]* 0.0174533
        lon2 = y_pred[:,1]* 0.0174533

        #haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = tf.sin(dlat/2)**2 + tf.cos(lat1) * tf.cos(lat2) * tf.sin(dlon/2)**2
        c = 2 * tf.asin(tf.sqrt(a))
        distance = 6371 * c

        #sum all distances from the current batch and do a cumulative sum with the previous total
        self.total.assign_add(tf.reduce_sum(distance)) #tf equivalent of +=
        self.count.assign_add(tf.cast(tf.size(distance), tf.float32)) #nb of element and cast it to a float32 format

    #return MAE, can be called after each batch for display or each epoch for looging
    def result(self):
        return self.total / self.count #average

    #called at beginning of each epoch. reset mae
    def reset_state(self):
        self.total.assign(0.)
        self.count.assign(0.)



#

# tensor flow haversine scorer
