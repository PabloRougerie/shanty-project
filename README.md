# 🚢 Shanty Project - Maritime Vessel Position Prediction

Predicting vessel positions 1 hour ahead using AIS (Automatic Identification System) data to estimate arrival times at ports.

## Objective

Given a vessel's trajectory over the past 60 minutes, predict its position 60 minutes into the future.

**Use case:** Estimate vessel arrival times for port congestion management and cargo tracking.

## Context

2-day technical project to demonstrate geospatial + time-series ML skills for maritime applications.

## Approach

**Data:** AIS vessel tracking (Gulf of Mexico, December 2024)
- Cargo and Tanker vessels
- 5 days of position reports (~15min intervals)

**Models:**
- **XGBoost:** Feature engineering approach (lagged positions, velocity, aggregates)
- **LSTM:** Deep learning sequence model

**Evaluation:**
- Mean Absolute Error (MAE) in kilometers
- Root Mean Square Error (RMSE)
- Haversine distance for accurate geospatial calculations

## Tech Stack

- **Geospatial:** `geopandas`, `folium`, Haversine distance
- **ML:** `scikit-learn`, `xgboost`
- **DL:** `tensorflow` (Metal GPU on M4)
- **Data:** `pandas`, `numpy`, `pyarrow`

## Project Structure
```
shanty-project/
├── data/               # AIS data (gitignored)
├── notebooks/          # EDA, XGBoost, LSTM
├── requirements.txt
└── README.md
```

## Results

Work in progress

## Author

Pablo Rougerie - [LinkedIn](https://linkedin.com/in/pablorougerie) | [GitHub](https://github.com/PabloRougerie)
