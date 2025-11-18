"""
Model training module for fruit classification.
Loads the dataset, preprocesses it, trains a model, and saves it.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os


def train_fruit_model(data_path='data/Fruit Classification Dataset.csv', model_path='model.pkl'):
    """
    Train a fruit classification model.
    
    Args:
        data_path: Path to the CSV dataset file
        model_path: Path where the trained model will be saved
    
    Returns:
        dict: Dictionary containing model metrics and label encoder
    """
    print("Loading dataset...")
    # Load the dataset
    df = pd.read_csv(data_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"First few rows:\n{df.head()}")
    
    # Identify target column (common names: 'label', 'fruit', 'class', 'target', or last column)
    target_candidates = ['label', 'fruit', 'class', 'target', 'Fruit', 'Label', 'Class']
    target_col = None
    
    for col in target_candidates:
        if col in df.columns:
            target_col = col
            break
    
    # If no standard target column found, use the last column
    if target_col is None:
        target_col = df.columns[-1]
        print(f"No standard target column found. Using last column: {target_col}")
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Handle categorical features in X
    label_encoders = {}
    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
    
    # Encode target variable if it's categorical
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)
    class_names = target_encoder.classes_
    
    print(f"\nTarget classes: {class_names}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Feature names: {X.columns.tolist()}")
    
    # Handle missing values
    if X.isnull().sum().sum() > 0:
        print("Handling missing values...")
        X = X.fillna(X.mean())
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Train the model
    print("\nTraining Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nModel Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Identify which features are categorical (have encoders)
    categorical_features = list(label_encoders.keys())
    
    # Save the model
    model_data = {
        'model': model,
        'target_encoder': target_encoder,
        'feature_encoders': label_encoders,
        'feature_names': X.columns.tolist(),
        'class_names': class_names.tolist(),
        'categorical_features': categorical_features
    }
    
    joblib.dump(model_data, model_path)
    print(f"\nModel saved to {model_path}")
    
    return {
        'accuracy': accuracy,
        'class_names': class_names.tolist(),
        'feature_names': X.columns.tolist()
    }


if __name__ == '__main__':
    # Train the model when run directly
    train_fruit_model()

