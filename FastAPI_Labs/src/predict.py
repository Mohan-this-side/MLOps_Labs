import joblib
import numpy as np
from data import get_target_names

def predict_data(X):
    """
    Predict the class labels for the input wine data with preprocessing.
    Args:
        X (numpy.ndarray): Input data for which predictions are to be made.
    Returns:
        dict: Dictionary containing prediction, class name, and confidence.
    """
    # Load the trained model and scaler
    model = joblib.load("../model/wine_model.pkl")
    scaler = joblib.load("../model/wine_scaler.pkl")
    
    # Scale the input data using the same scaler used during training
    X_scaled = scaler.transform(X)
    
    # Make prediction
    y_pred = model.predict(X_scaled)
    
    # Get prediction probabilities for confidence
    y_proba = model.predict_proba(X_scaled)
    confidence = np.max(y_proba, axis=1)
    
    # Get class names
    target_names = get_target_names()
    
    # Return detailed prediction information
    results = []
    for i in range(len(y_pred)):
        results.append({
            'prediction': int(y_pred[i]),
            'class_name': target_names[y_pred[i]],
            'confidence': float(confidence[i])
        })
    
    return results[0] if len(results) == 1 else results

def get_wine_features_info():
    """
    Get information about wine dataset features for API documentation.
    Returns:
        dict: Dictionary with feature names and descriptions.
    """
    feature_info = {
        'alcohol': 'Alcohol percentage',
        'malic_acid': 'Malic acid content',
        'ash': 'Ash content',
        'alcalinity_of_ash': 'Alcalinity of ash',
        'magnesium': 'Magnesium content',
        'total_phenols': 'Total phenols',
        'flavanoids': 'Flavanoids content',
        'nonflavanoid_phenols': 'Non-flavanoid phenols',
        'proanthocyanins': 'Proanthocyanins content',
        'color_intensity': 'Color intensity',
        'hue': 'Hue value',
        'od280_od315_of_diluted_wines': 'OD280/OD315 ratio of diluted wines',
        'proline': 'Proline content'
    }
    return feature_info
