import pandas as pd


def checks(df: pd.DataFrame):
    # Null Values
    na_values = df.isna().sum().sum()
    print(f"Null Values found: {na_values > 0}")
    print(f"Count of Null Values: {na_values}")

    # Duplicate Rows
    if df[df.duplicated(keep=False)].empty:
        print("Duplicate Rows: None")
    else:
        print("Duplicate Rows")
        for _, row in df[df.duplicated(keep=False)].iterrows():
            print(row)
        df.drop_duplicates(inplace=True)

    # Finding the range of each feature in the dataset
    ranges = {}
    for col in df.columns:
        ranges[col] = f"{df[col].min()}-{df[col].max()}"

    print("Range of each feature:")
    for key, val in ranges.items():
        print(f"{key}: {val}")
