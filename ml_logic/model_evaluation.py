import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.multioutput import MultiOutputRegressor

from ml_logic.feature_engineering import create_time_series_features
from ml_logic.metric import position_extrapolation, haversine_mae

import tensorflow as tf

import matplotlib.pyplot as plt

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



#Function to draw the learning curve
def plot_training_history(history):

    fig, axes = plt.subplots(1,2, figsize=(14,5))

    #plot Loss (Huber)
    axes[0].plot(history.history['loss'], label= "Train Loss")
    axes[0].plot(history.history['val_loss'], label= 'Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Huber Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    #Haversine MAE
    axes[1].plot(history.history['haversine_mae_km'], label='Train MAE')
    axes[1].plot(history.history['val_haversine_mae_km'], label='Val MAE')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Haversine MAE (km)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    plt.show()


#

# tensor flow haversine scorer
