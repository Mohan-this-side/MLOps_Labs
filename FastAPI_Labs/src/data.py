import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data():
    """
    Load the Wine dataset and return the features and target values.
    Returns:
        X (numpy.ndarray): The features of the Wine dataset (13 features).
        y (numpy.ndarray): The target values of the Wine dataset (3 classes).
    """
    wine = load_wine()
    X = wine.data
    y = wine.target
    return X, y

def get_feature_names():
    """
    Get the feature names for the Wine dataset.
    Returns:
        feature_names (list): List of feature names.
    """
    wine = load_wine()
    return wine.feature_names.tolist()

def get_target_names():
    """
    Get the target class names for the Wine dataset.
    Returns:
        target_names (list): List of target class names.
    """
    wine = load_wine()
    return wine.target_names.tolist()

def split_data(X, y):
    """
    Split the data into training and testing sets with feature scaling.
    Args:
        X (numpy.ndarray): The features of the dataset.
        y (numpy.ndarray): The target values of the dataset.
    Returns:
        X_train, X_test, y_train, y_test (tuple): The split and scaled dataset.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Scale the features for better model performance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test