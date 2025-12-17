# 🚢 Shanty Project - Maritime Vessel Position Prediction

Predicting vessel positions across multiple time horizons (1h to 36h) using AIS (Automatic Identification System) data to improve maritime traffic management and port operations.

## Objective

Given a vessel's current position, speed (SOG), and course (COG), predict its position at various future time horizons (30 minutes to 36 hours).

**Use case:**
- Estimate vessel arrival times for port congestion management
- Improve collision avoidance systems
- Enhance search and rescue operations
- Optimize route planning

## Context

Short technical project demonstrating geospatial + time-series ML skills for maritime applications. The project systematically evaluates when machine learning models provide value over simple linear extrapolation.

## Approach

**Data:** AIS vessel tracking (Gulf of Mexico, December 2024)
- Cargo and Tanker vessels
- 5 days of position reports (resampled to 5-minute intervals)
- ~1 million pings from ~1000 vessels

**Methodology:**
1. **Data Preprocessing**: Cleaning, deduplication, resampling to uniform 5-minute intervals
2. **Baseline Model**: Linear extrapolation using current position, SOG, and COG
3. **Feature Engineering**: Lagged features (LAT, LON, SOG, COG) proportional to prediction horizon
4. **Model Comparison**: Multiple algorithms across 19 prediction horizons (1h to 36h)
5. **Feature Selection**: Permutation importance to identify redundant features
6. **Hyperparameter Tuning**: GridSearchCV for optimal model parameters

**Models Tested:**
- **Baseline**: Linear extrapolation (naive baseline)
- **Linear Models**: Linear Regression, Ridge, Lasso
- **Tree-based**: XGBoost, LightGBM, Random Forest
- **Final Selection**: Ridge (scaled) and LightGBM (tuned)

**Evaluation:**
- Mean Absolute Error (MAE) in kilometers using Haversine distance
- GroupShuffleSplit for train/test (vessel-level splitting to prevent data leakage)
- GroupKFold for cross-validation
- Custom Haversine MAE scorer

## Key Findings

### Three Prediction Regimes

1. **Short-term (< 3h)**: Baseline wins
   - Linear extrapolation sufficient (MAE 1-7 km)
   - ML models underperform due to overfitting

2. **Medium-term (3-7h)**: ML becomes useful
   - LightGBM starts outperforming baseline at ~4 hours
   - Improvement: 3-5 km (15-20% better than baseline)

3. **Long-term (7-36h)**: ML models much better
   - LightGBM (tuned): 13-18 km improvement at 14-21h horizons
   - 20-30% improvement over baseline
   - Ridge (scaled) also performs well, offering simpler alternative

### Real-World Impact

At 14-21 hour prediction horizons:
- **Error reduction**: 13-18 km improvement over baseline
- **Operational significance**: Better collision avoidance, more accurate ETAs, improved search and rescue positioning

### Feature Engineering & Selection

- **Rolling features**: No improvement (tree models can implicitly capture patterns)
- **Advanced features**: No improvement (multicollinearity, linear problem nature)
- **Feature selection**: Removed 3 low-importance features (`COG_lag_240min`, `COG_lag_480min`, `SOG_lag_240min`) with minimal but positive impact

## Tech Stack

- **Geospatial**: Haversine distance calculations
- **ML**: `scikit-learn`, `lightgbm`, `xgboost`
- **Data**: `pandas`, `numpy`, `pyarrow`
- **Visualization**: `matplotlib`, `seaborn`

## Project Structure

```
shanty-project/
├── data/
│   ├── raw/              # Original AIS data
│   └── processed/        # Cleaned and resampled data
├── ml_logic/             # Modular ML functions
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_evaluation.py
│   └── metric.py
├── notebooks/            # Analysis notebooks
│   ├── 1_exploratory_analysis.ipynb
│   ├── 2_horizon_comparison.ipynb
│   ├── 3_feature_engineering.ipynb
│   ├── 4_feature_selection.ipynb
│   ├── 5_model_tuning.ipynb
│   └── 6_final_comparison.ipynb
├── visualizations/       # Generated plots
├── requirements.txt
└── README.md
```

## Results

**Best Model**: LightGBM (tuned) for horizons > 4 hours

**Performance Highlights**:
- 7h horizon: 3.1 km improvement (19% better)
- 14h horizon: 13.2 km improvement (30.5% better)
- 21h horizon: 18.2 km improvement (28.9% better)

**Key Insight**: Machine learning provides meaningful value only for prediction horizons beyond 3-4 hours. For shorter horizons, simple linear extrapolation is sufficient and more reliable.

## Author

Pablo Rougerie - [LinkedIn](https://linkedin.com/in/pablorougerie) | [GitHub](https://github.com/PabloRougerie)
