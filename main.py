from pathlib import Path
from data_processing import process_data
from model import train_and_evaluate
from eval import create_submission


def main():
    # Define paths
    base_path = Path("bike_counters_Drion_Filisetti")
    train_path = base_path / "data" / "train.parquet"
    test_path = base_path / "data" / "final_test.parquet"
    external_data_path = base_path / "external_data" / "external_data.csv"
    submission_path = base_path / "submission.csv"

    # Process data
    print("Processing data...")
    processed_data = process_data(
        train_path=str(train_path),
        test_path=str(test_path),
        external_data_path=str(external_data_path)
    )

    # Prepare training data
    train_data = processed_data["train"]
    X = train_data.drop("log_bike_count", axis=1)
    y = train_data["log_bike_count"]

    # Train and evaluate model
    print("\nTraining and evaluating model...")
    model = train_and_evaluate(X, y)

    # Create submission
    print("\nCreating submission...")
    create_submission(model, processed_data["test"], submission_path)


if __name__ == "__main__":
    main()
