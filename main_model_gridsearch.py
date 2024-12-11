import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
from data_processing import (
    load_and_clean_external_data,
    add_covid_restrictions_holiday,
    expand_hourly_data,
    add_temporal_features,
    set_headings
)


def load_test_data(test_path, external_data_path):
    """
    Load and process test data, combining it with external data and adding features.

    Args:
        test_path: Path to the test parquet file
        external_data_path: Path to the external data CSV file

    Returns:
        processed_data: DataFrame ready for prediction
        original_index: Series mapping processed data back to original order
    """
    # Load test data and store original index
    test_data = pd.read_parquet(test_path)
    original_index = pd.Series(range(len(test_data)), index=test_data.index)

    # Process external data
    external_data = load_and_clean_external_data(external_data_path)
    external_data = add_covid_restrictions_holiday(external_data)
    external_data = set_headings(external_data)
    external_data = expand_hourly_data(external_data)

    # Merge datasets
    final_data = pd.merge(test_data, external_data, on="date", how="inner")

    # Add temporal features
    final_data = add_temporal_features(final_data)

    # Drop unnecessary columns
    columns_to_drop = [
        "counter_name",
        "site_id",
        "site_name",
        "date",
        "counter_installation_date",
        "coordinates",
        "counter_technical_id",
        "season",
    ]
    final_data.drop(columns=columns_to_drop, inplace=True)

    return final_data, original_index


def make_predictions(model_path, X_test):
    """
    Load model and make predictions.

    Args:
        model_path: Path to the saved model file
        X_test: DataFrame of features to predict on

    Returns:
        numpy array of predictions
    """
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    return model.predict(X_test)


def create_submission(predictions, ids, output_path):
    """
    Create and save submission file.

    Args:
        predictions: Array of predicted values
        ids: Series of ID values
        output_path: Path to save the submission file
    """
    submission = pd.DataFrame({"Id": ids, "log_bike_count": predictions}).sort_values(
        "Id"
    )

    submission.to_csv(output_path, index=False)
    return submission


def main():
    # Define paths
    BASE_PATH = Path("bike_counters_Drion_Filisetti")
    TEST_PATH = BASE_PATH / "data" / "final_test.parquet"
    EXTERNAL_DATA_PATH = BASE_PATH / "external_data" / "external_data.csv"
    MODEL_PATH = BASE_PATH / "outputs" / "model_optimized.json"
    OUTPUT_PATH = BASE_PATH / "submission.csv"

    # Load and process data
    print("Loading and processing data...")
    test_processed, original_index = load_test_data(TEST_PATH, EXTERNAL_DATA_PATH)

    # Prepare features
    feature_cols = [col for col in test_processed.columns if col != "Id"]
    X_test = test_processed[feature_cols]

    # Make predictions
    print("Making predictions...")
    predictions = make_predictions(MODEL_PATH, X_test)

    # Create submission file
    print("Creating submission file...")
    submission = create_submission(
        predictions, test_processed.index.map(original_index), OUTPUT_PATH
    )

    # Verify submission
    print("\nSubmission verification:")
    print(f"Number of predictions: {len(submission)}")
    print(f"ID range: {submission['Id'].min()} to {submission['Id'].max()}")
    print(
        f"Any missing IDs: {not submission['Id'].equals(pd.Series(range(len(submission))))}"
    )
    print(f"Submission saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
