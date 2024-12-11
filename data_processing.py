import numpy as np
import pandas as pd
from pathlib import Path

def load_and_clean_external_data(filepath):
    """Load and clean external data, handling missing values and date formatting."""
    external_data = pd.read_csv(filepath)
    external_data_cleaned = external_data.dropna(axis=1, how='all')

    columns_of_interest = ["date", "etat_sol", "dd", "ff", "t", "u", "vv", "n", "ht_neige", "rr3"]
    external_data_sorted = external_data_cleaned[columns_of_interest].copy()
    external_data_sorted["date"] = pd.to_datetime(external_data_sorted['date'])

    # Convert temperature from Kelvin to Celsius
    external_data_sorted.loc[:, "t"] = external_data_sorted["t"] - 273.15

    return external_data_sorted

def add_covid_restrictions_holiday(df):
    """Add COVID-related restriction and holiday periods as features."""
    periods = {
        'Lockdown': [
            ('2020-10-30', '2020-12-15'),
            ('2021-04-03', '2021-05-04')
        ],
        'soft-curfew': [
            ('2020-10-17', '2020-10-30'),
            ('2020-12-15', '2021-01-16'),
            ('2021-05-19', '2021-06-21')
        ],
        'hard-curfew': [
            ('2021-01-16', '2021-04-03'),
            ('2021-05-04', '2021-05-19')
        ],
        'holidays' : [
            ('2020-10-17', '2020-11-02'),
            ('2020-12-19', '2021-01-02'),
            ('2021-02-13', '2021-03-01'),
            ('2021-04-17', '2021-05-03'),
            ('2021-10-23', '2021-11-08')
        ]
    }

    for type, periods in periods.items():
        df[type] = 0
        for start_date, end_date in periods:
            mask = (df['date'] >= start_date) & (df['date'] < end_date)
            df.loc[mask, type] = 1

    return df

def set_headings(df):
    column_names = ["East", "South", "West"]
    bearings = [(45, 135), (135, 225), (225, 315)]

    for column_name, (low_bearing, high_bearing) in zip(column_names, bearings):
        df[column_name] = 0
        df.loc[(df['dd'] >= low_bearing) & (df['dd'] < high_bearing), column_name] = 1

    return df.drop(columns=["dd"])

def expand_hourly_data(df):
    """Create missing hourly data points by copying existing rows."""
    def create_missing_hours(row):
        return [
            {**row.to_dict(), 'date': row['date'] - pd.Timedelta(hours=h)}
            for h in [2, 1]
        ]

    new_rows = []
    for _, row in df.iterrows():
        new_rows.extend(create_missing_hours(row))

    expanded_df = pd.concat([
        df,
        pd.DataFrame(new_rows)
    ], ignore_index=True)

    return expanded_df.sort_values(by='date').reset_index(drop=True)

def add_temporal_features(df):
    """Add various temporal features that might be useful for prediction."""
    df = df.copy()

    # Basic time features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['weekday'] = df['date'].dt.weekday
    df['hour'] = df['date'].dt.hour

    # Additional useful time features
    df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)
    holidays = pd.to_datetime(['2020-11-01', '2020-11-11', '2020-12-25', '2021-01-01', '2021-04-05',
                               '2021-05-01', '2021-05-13', '2021-05-24', '2021-07-14', '2021-08-15', '2021-11-01', '2021-11-11'])
    df['is_holiday'] = df['date'].isin(holidays).astype(int)
    df['season'] = df['month'].map(lambda m: (m%12 + 3)//3)

    # One-hot encoding for counter_id
    df = pd.get_dummies(df, columns=["counter_id"], dummy_na=True, drop_first=True, prefix_sep=' ')
    df.drop(columns=["counter_id nan"], inplace=True)

    return df

def process_data(train_path=None, test_path=None, external_data_path=None):
    """Process either training or test data."""
    # Load external data
    external_data = load_and_clean_external_data(external_data_path)
    external_data = add_covid_restrictions_holiday(external_data)
    external_data = expand_hourly_data(external_data)
    external_data = set_headings(external_data)

    result = {}

    if train_path:
        # Process training data
        train_data = pd.read_parquet(train_path)
        train_merged = pd.merge(train_data, external_data, on='date', how='inner')
        train_processed = add_temporal_features(train_merged)
        train_processed = train_processed.sort_values(['date', 'counter_name'])

        # Drop unnecessary columns
        train_processed.drop(columns=['counter_name', 'site_id', 'site_name', 'bike_count',
                                    'date', 'counter_installation_date', 'coordinates',
                                    'counter_technical_id', 'season'], inplace=True)
        result['train'] = train_processed

    if test_path:
        # Process test data
        test_data = pd.read_parquet(test_path)
        original_index = pd.Series(range(len(test_data)), index=test_data.index)

        test_merged = pd.merge(test_data, external_data, on='date', how='inner')
        test_processed = add_temporal_features(test_merged)

        # Drop unnecessary columns
        test_processed.drop(columns=['counter_name', 'site_id', 'site_name',
                                   'date', 'counter_installation_date', 'coordinates',
                                   'counter_technical_id', 'season'], inplace=True)

        test_processed['Id'] = original_index[test_processed.index]
        result['test'] = test_processed

    return result
