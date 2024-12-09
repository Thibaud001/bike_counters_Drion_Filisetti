import pandas as pd

def create_submission(model, test_data, submission_path):
    """Create and save submission file."""
    feature_cols = [col for col in test_data.columns if col != 'Id']
    predictions = model.predict(test_data[feature_cols])
    
    submission = pd.DataFrame({
        'Id': test_data['Id'],
        'log_bike_count': predictions
    }).sort_values('Id')
    
    submission.to_csv(submission_path, index=False)
    
    # Verify submission
    print(f"\nSubmission stats:")
    print(f"Number of predictions: {len(submission)}")
    print(f"ID range: {submission['Id'].min()} to {submission['Id'].max()}")
    print(f"Any missing IDs: {not submission['Id'].equals(pd.Series(range(len(submission))))}")