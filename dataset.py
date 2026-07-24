"""
This file is meant to be imported by Erin's main.py during integration:
    from dataset import load_student_dataset
    X, y, feature_names = load_student_dataset()
"""

import pandas as pd
import numpy as np


def load_student_dataset(filename='student_dataset.csv'):
    """
    Loads the student dataset from a CSV file.

    Args:
        filename (str): Path to the CSV file. Defaults to 'student_dataset.csv'.

    Returns:
        X (numpy array): Shape (10, 3) -> the 3 input features
                          (Study_Hours, Assignments, Classes_Missed)
        y (numpy array): Shape (10,)   -> the target variable (1 = Pass, 0 = Fail)
        feature_names (list): Column names for the 3 features, in order.
                               Isbat needs this to label the input circles
                               in the diagram (e.g. "Study_Hours" instead of "X1").
    """

    # 1. Load the CSV file using Pandas
    # We use Pandas because it neatly organizes the tabular data and
    # makes it easy to separate features from the target column by name.
    df = pd.read_csv(filename)

    # 2. Separate the Input Features from the Target
    # We drop the 'Target_Pass' column to keep only our 3 input features.
    features = df.drop(columns=['Target_Pass'])

    # We isolate the 'Target_Pass' column to act as our expected output (y).
    target = df['Target_Pass']

    # 3. Keep the feature column names before converting to NumPy.
    # Pandas remembers column names; NumPy arrays don't. We save the names
    # here so the visualization step can still label each input node correctly.
    feature_names = features.columns.tolist()

    # 4. Convert Pandas DataFrames to NumPy Arrays.
    # Neural network math (dot products for summation) requires NumPy
    # arrays, not Pandas DataFrames.
    X = features.to_numpy()
    y = target.to_numpy()

    # 5. Print a quick summary so anyone running this file standalone
    # can confirm the dataset loaded correctly before building on top of it.
    print("Dataset successfully loaded!")
    print(f"Total Samples: {len(X)}")
    print(f"Features per sample: {X.shape[1]}")
    print(f"Feature names: {feature_names}")
    print("-" * 30)

    return X, y, feature_names


# Quick standalone test so Farzana, Isbat, and Erin can trust this file
# works correctly before they build their own code on top of it.
if __name__ == "__main__":
    X, y, feature_names = load_student_dataset()

    # Print the first student to visually verify the data loaded right
    print("First student inputs (X[0]):", X[0])
    print("First student target  (y[0]):", y[0])

    # Print one Pass and one Fail example so the team can eyeball
    # that the labels make sense against the raw numbers
    pass_index = np.where(y == 1)[0][0]
    fail_index = np.where(y == 0)[0][0]
    print(f"\nExample PASS student -> {dict(zip(feature_names, X[pass_index]))}")
    print(f"Example FAIL student -> {dict(zip(feature_names, X[fail_index]))}")