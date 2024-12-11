import optuna
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import logging
from data_processing import (
    load_and_clean_external_data,
    add_covid_restrictions_holiday,
    expand_hourly_data,
    add_temporal_features,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_data():
    """Prepare the data using the existing processing pipeline."""
    # Load external data
    external_data = load_and_clean_external_data("bike_counters_Drion_Filisetti/external_data/external_data.csv")
    
    # Add COVID restrictions
    external_data = add_covid_restrictions_holiday(external_data)
    
    # Expand to hourly data
    external_data = expand_hourly_data(external_data)
    
    # Load training data
    train_data = pd.read_parquet("bike_counters_Drion_Filisetti/data/train.parquet")
    
    # Merge datasets
    final_data = pd.merge(train_data, external_data, on='date', how='inner')
    
    # Add temporal features
    final_data = add_temporal_features(final_data)
    
    # Sort by date and counter
    final_data = final_data.sort_values(['date', 'counter_name'])

    # Drop columns that are not needed
    final_data.drop(columns=['counter_name', 'site_id', 'site_name', 'bike_count', 'date', 
                           'counter_installation_date', 'coordinates', 'counter_technical_id', 
                           'season'], inplace=True)
    
    return final_data

def objective(trial):
    # Load and process data
    logger.info("Loading and processing data...")
    data_processed = prepare_data()
    
    feature_cols = [col for col in data_processed.columns if col != 'log_bike_count']
    X = data_processed[feature_cols]
    y = data_processed['log_bike_count']
    
    # Define hyperparameter search space
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
    }
    
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    scores = []
    
    # Perform time series cross-validation
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Initialize model
        model = XGBRegressor(
            **param,
            random_state=42,
            n_jobs=-1,  # Use all available cores
            callbacks=[xgboost.callback.EarlyStopping(rounds=50)]
        )
        
        # Fit with evaluation set
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # Predict and evaluate
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        scores.append(rmse)
        
        logger.info(f"Trial {trial.number} Fold {fold+1} RMSE: {rmse:.4f}")
    
    # Calculate mean RMSE across folds
    mean_rmse = np.mean(scores)
    logger.info(f"Trial {trial.number} Mean RMSE: {mean_rmse:.4f}")
    
    return mean_rmse

def run_optimization(n_trials=100):
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
        study_name="xgboost_optimization"
    )
    
    study.optimize(objective, n_trials=n_trials)
    
    logger.info("Best trial:")
    trial = study.best_trial
    
    logger.info(f"  Value (RMSE): {trial.value:.4f}")
    logger.info("  Params: ")
    for key, value in trial.params.items():
        logger.info(f"    {key}: {value}")
    
    # Train final model with best parameters
    final_model = train_final_model(trial.params)
    
    return study, final_model

def train_final_model(best_params):
    # Load and process data
    logger.info("Loading and processing data for final model...")
    data_processed = prepare_data()
    
    feature_cols = [col for col in data_processed.columns if col != 'log_bike_count']
    X = data_processed[feature_cols]
    y = data_processed['log_bike_count']
    
    # Train final model
    final_model = XGBRegressor(
        **best_params,
        random_state=42,
        n_jobs=-1,
        callbacks=[xgboost.callback.EarlyStopping(rounds=50)]
    )
    
    final_model.fit(X, y)
    
    # Save the model
    final_model.save_model('bike_counters_Drion_Filisetti/model_optimized.json')
    
    # Save feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    feature_importance.to_csv('bike_counters_Drion_Filisetti/feature_importance.csv', index=False)
    
    return final_model

if __name__ == "__main__":
    # Import xgboost here to use callback
    import xgboost
    study, final_model = run_optimization(n_trials=100)
    
    # Save study results
    study_results = pd.DataFrame({
        'number': [t.number for t in study.trials],
        'value': [t.value for t in study.trials],
        **{f'param_{k}': [t.params.get(k) for t in study.trials] 
           for k in study.best_trial.params.keys()}
    })
    study_results.to_csv('bike_counters_Drion_Filisetti/optimization_results.csv', index=False)