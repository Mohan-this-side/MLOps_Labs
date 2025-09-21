from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib
import numpy as np
from data import load_data, split_data, get_target_names

def fit_model(X_train, y_train, X_test, y_test):
    """
    Train a Random Forest Classifier and save the model with scaler to files.
    Args:
        X_train (numpy.ndarray): Training features (already scaled).
        y_train (numpy.ndarray): Training target values.
        X_test (numpy.ndarray): Test features (already scaled).
        y_test (numpy.ndarray): Test target values.
    """
    # Train Random Forest Classifier (better for wine dataset)
    rf_classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    rf_classifier.fit(X_train, y_train)
    
    # Make predictions and evaluate
    y_pred = rf_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Print model performance
    print(f"Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    target_names = get_target_names()
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # Save the trained model
    joblib.dump(rf_classifier, "../model/wine_model.pkl")
    print("Model saved as 'wine_model.pkl'")
    
    return rf_classifier

def save_scaler_for_api():
    """
    Create and save the scaler that will be used in the API for preprocessing.
    This ensures consistent scaling between training and prediction.
    """
    X, y = load_data()
    scaler = StandardScaler()
    scaler.fit(X)  # Fit on the entire dataset
    joblib.dump(scaler, "../model/wine_scaler.pkl")
    print("Scaler saved as 'wine_scaler.pkl'")

if __name__ == "__main__":
    print("Loading Wine Dataset...")
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    print("Training Random Forest Classifier...")
    model = fit_model(X_train, y_train, X_test, y_test)
    
    print("Saving scaler for API...")
    save_scaler_for_api()
