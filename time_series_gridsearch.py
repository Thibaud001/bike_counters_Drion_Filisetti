import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterGrid
import logging
from datetime import datetime
from data_processing import (
    load_and_clean_external_data,
    add_covid_restrictions_holiday,
    expand_hourly_data,
    add_temporal_features,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

def load_data():
    """Load and prepare data for modeling using the processing pipeline."""
    logging.info("Loading and processing data...")
    
    # Load training data
    train_data = pd.read_parquet("bike_counters_Drion_Filisetti/data/train.parquet")
    
    # Process external data
    external_data = load_and_clean_external_data("bike_counters_Drion_Filisetti/external_data/external_data.csv")
    external_data = add_covid_restrictions_holiday(external_data)
    external_data = expand_hourly_data(external_data)
    
    # Merge datasets
    final_data = pd.merge(train_data, external_data, on='date', how='inner')
    
    # Add temporal features
    final_data = add_temporal_features(final_data)
    
    # Sort by date and counter
    final_data = final_data.sort_values(['date', 'counter_name'])
    
    # Store date column for time-based splitting
    dates = final_data['date']
    
    # Drop columns that are not needed
    final_data.drop(columns=['counter_name', 'site_id', 'site_name', 'bike_count', 'date', 
                            'counter_installation_date', 'coordinates', 'counter_technical_id', 'season'], 
                   inplace=True)
    
    # Separate features and target
    y = final_data['log_bike_count']
    X = final_data.drop('log_bike_count', axis=1)
    feature_cols = X.columns.tolist()
    
    return X, y, feature_cols, dates

def time_series_cv_score(model, X, y, dates, n_splits=5):
    """Perform time series cross-validation and return scores."""
    # Create splits based on dates
    unique_dates = dates.unique()
    split_size = len(unique_dates) // (n_splits + 1)
    scores = []
    
    for fold in range(n_splits):
        split_date = unique_dates[split_size * (fold + 1)]
        
        # Split data based on dates
        train_mask = dates < split_date
        test_mask = (dates >= split_date) & (dates < unique_dates[split_size * (fold + 2)])
        
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        
        if len(X_train) == 0 or len(X_test) == 0:
            continue
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        scores.append(rmse)
        
        logging.info(f"Fold {fold + 1} RMSE: {rmse:.4f}")
    
    return np.mean(scores), np.std(scores)

def grid_search_cv():
    """Perform grid search with time series cross-validation."""
    X, y, feature_cols, dates = load_data()
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [500, 1000, 1500],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [5, 7, 9],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'min_child_weight': [1, 3, 5]
    }
    
    # Initialize tracking of best parameters
    best_score = float('inf')
    best_params = None
    best_std = None
    results = []
    
    # Perform grid search
    for params in ParameterGrid(param_grid):
        logging.info(f"\nTesting parameters: {params}")
        
        model = XGBRegressor(
            **params,
            random_state=42,
            n_jobs=-1  # Use all available cores
        )
        
        mean_rmse, std_rmse = time_series_cv_score(model, X, y, dates)
        
        results.append({
            **params,
            'mean_rmse': mean_rmse,
            'std_rmse': std_rmse
        })
        
        logging.info(f"Mean RMSE: {mean_rmse:.4f} (+/- {std_rmse:.4f})")
        
        if mean_rmse < best_score:
            best_score = mean_rmse
            best_params = params
            best_std = std_rmse
    
    # Save results to CSV
    pd.DataFrame(results).to_csv('grid_search_results.csv', index=False)
    
    return best_params, best_score, best_std, X, y, feature_cols

def train_final_model(best_params, X, y, feature_cols):
    """Train final model with best parameters and save it."""
    logging.info("\nTraining final model with best parameters...")
    
    final_model = XGBRegressor(**best_params, random_state=42, n_jobs=-1)
    final_model.fit(X, y)
    
    # Save model
    final_model.save_model('bike_counters_Drion_Filisetti/model.json')
    
    # Get and save feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    feature_importance.to_csv('feature_importance.csv', index=False)
    
    logging.info("\nTop 10 important features:")
    logging.info(feature_importance.head(10))
    
    return final_model

def main():
    logging.info("Starting model training process...")
    
    # Perform grid search
    best_params, best_score, best_std, X, y, feature_cols = grid_search_cv()
    
    logging.info("\nBest parameters found:")
    logging.info(f"Parameters: {best_params}")
    logging.info(f"Best RMSE: {best_score:.4f} (+/- {best_std:.4f})")
    
    # Train final model
    final_model = train_final_model(best_params, X, y, feature_cols)
    
    logging.info("Model training completed!")

if __name__ == "__main__":
    main()