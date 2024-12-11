import optuna
import pandas as pd
import numpy as np
import xgboost
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import logging
from pathlib import Path
from data_processing import process_data

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BikeCountOptimizer:
    def __init__(self):
        self.output_dir = Path("bike_counters_Drion_Filisetti/outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def prepare_data(self):
        """Prepare the data using the current processing pipeline."""
        logger.info("Processing training data...")
        
        # Define paths
        train_path = Path("bike_counters_Drion_Filisetti/data/train.parquet")
        external_data_path = Path("bike_counters_Drion_Filisetti/external_data/external_data.csv")
        
        # Process training data using the unified process_data function
        result = process_data(
            train_path=train_path,
            external_data_path=external_data_path
        )
        
        # Extract training data
        processed_data = result['train']
        
        return processed_data

    def objective(self, trial):
        # Load and process data
        logger.info("Loading and processing data...")
        processed_data = self.prepare_data()
        
        # Prepare features and target
        y = processed_data['log_bike_count']
        X = processed_data.drop(['log_bike_count'], axis=1)
        
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
        tscv = TimeSeriesSplit(n_splits=5, test_size=int(len(X) * 0.1))
        scores = []
        
        # Perform time series cross-validation
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Initialize model
            model = XGBRegressor(
                **param,
                random_state=42,
                n_jobs=-1,
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
            
            # Log details about the split
            logger.info(f"Trial {trial.number} Fold {fold+1}")
            logger.info(f"Training set size: {len(X_train)}")
            logger.info(f"Test set size: {len(X_test)}")
            logger.info(f"RMSE: {rmse:.4f}")
        
        # Calculate mean RMSE across folds
        mean_rmse = np.mean(scores)
        logger.info(f"Trial {trial.number} Mean RMSE: {mean_rmse:.4f}")
        
        return mean_rmse

    def train_final_model(self, best_params):
        # Load and process data
        logger.info("Loading and processing data for final model...")
        processed_data = self.prepare_data()
        
        # Prepare features and target
        y = processed_data['log_bike_count']
        X = processed_data.drop(['log_bike_count'], axis=1)
        
        # Split data for final validation (last 10% as validation)
        train_size = int(len(X) * 0.9)
        X_train = X.iloc[:train_size]
        y_train = y.iloc[:train_size]
        X_val = X.iloc[train_size:]
        y_val = y.iloc[train_size:]
        
        # Train final model
        final_model = XGBRegressor(
            **best_params,
            random_state=42,
            n_jobs=-1,
            callbacks=[xgboost.callback.EarlyStopping(rounds=50)]
        )
        
        final_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=True
        )
        
        # Evaluate on validation set
        val_pred = final_model.predict(X_val)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        logger.info(f"Final model validation RMSE: {val_rmse:.4f}")
        
        # Save the model
        final_model.save_model(self.output_dir / 'model_optimized.json')
        
        # Save feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': final_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        feature_importance.to_csv(self.output_dir / 'feature_importance.csv', index=False)
        
        return final_model

    def run_optimization(self, n_trials=100):
        study = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
            study_name="xgboost_optimization"
        )
        
        study.optimize(lambda trial: self.objective(trial), n_trials=n_trials)
        
        logger.info("Best trial:")
        trial = study.best_trial
        
        logger.info(f"  Value (RMSE): {trial.value:.4f}")
        logger.info("  Params: ")
        for key, value in trial.params.items():
            logger.info(f"    {key}: {value}")
        
        # Train final model with best parameters
        final_model = self.train_final_model(trial.params)
        
        # Save study results
        study_results = pd.DataFrame({
            'number': [t.number for t in study.trials],
            'value': [t.value for t in study.trials],
            **{f'param_{k}': [t.params.get(k) for t in study.trials] 
               for k in study.best_trial.params.keys()}
        })
        study_results.to_csv(self.output_dir / 'optimization_results.csv', index=False)
        
        return study, final_model

if __name__ == "__main__":
    optimizer = BikeCountOptimizer()
    study, final_model = optimizer.run_optimization(n_trials=100)