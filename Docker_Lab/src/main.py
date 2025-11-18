"""
Flask web service for fruit classification predictions.
"""
from flask import Flask, request, jsonify, render_template
import joblib
import os
import numpy as np
from train_model import train_fruit_model

app = Flask(__name__, template_folder='templates')

# Global variables for model and metadata
model_data = None
model_loaded = False


def load_model():
    """Load the trained model or train it if it doesn't exist."""
    global model_data, model_loaded
    
    model_path = 'model.pkl'
    data_path = 'data/Fruit Classification Dataset.csv'
    
    # Check if model exists, if not train it
    if not os.path.exists(model_path):
        print("Model not found. Training new model...")
        try:
            train_fruit_model(data_path=data_path, model_path=model_path)
        except Exception as e:
            print(f"Error training model: {e}")
            return False
    
    # Load the model
    try:
        print("Loading trained model...")
        model_data = joblib.load(model_path)
        model_loaded = True
        print("Model loaded successfully!")
        print(f"Available classes: {model_data['class_names']}")
        print(f"Feature names: {model_data['feature_names']}")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False


@app.route('/')
def home():
    """Welcome page."""
    return "Welcome to the Fruit Classification API! Visit /predict to make predictions."


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Handle prediction requests."""
    global model_data, model_loaded
    
    # Ensure model is loaded
    if not model_loaded:
        if not load_model():
            return jsonify({"error": "Model could not be loaded. Please check the logs."}), 500
    
    if request.method == 'GET':
        # Render the prediction form
        categorical_features = model_data.get('categorical_features', [])
        # Get unique values for categorical features to create dropdowns
        categorical_options = {}
        for feature in categorical_features:
            if feature in model_data.get('feature_encoders', {}):
                encoder = model_data['feature_encoders'][feature]
                # Get original class names (before encoding)
                categorical_options[feature] = encoder.classes_.tolist()
        
        return render_template('predict.html', 
                             feature_names=model_data['feature_names'],
                             class_names=model_data['class_names'],
                             categorical_features=categorical_features,
                             categorical_options=categorical_options)
    
    elif request.method == 'POST':
        try:
            # Get form data
            if request.is_json:
                data = request.json
            else:
                data = request.form
            
            # Extract features in the correct order
            features = []
            feature_names = model_data['feature_names']
            
            for feature_name in feature_names:
                if feature_name in data:
                    value = data[feature_name]
                    # Try to convert to float
                    try:
                        features.append(float(value))
                    except ValueError:
                        # If conversion fails, check if it needs encoding
                        if feature_name in model_data.get('feature_encoders', {}):
                            encoder = model_data['feature_encoders'][feature_name]
                            try:
                                # Try to transform the value
                                encoded = encoder.transform([str(value)])[0]
                                features.append(float(encoded))
                            except:
                                return jsonify({
                                    "error": f"Invalid value '{value}' for feature '{feature_name}'"
                                }), 400
                        else:
                            return jsonify({
                                "error": f"Could not convert '{value}' to number for feature '{feature_name}'"
                            }), 400
                else:
                    return jsonify({
                        "error": f"Missing required feature: {feature_name}"
                    }), 400
            
            # Convert to numpy array and reshape for prediction
            features_array = np.array(features).reshape(1, -1)
            
            # Make prediction
            prediction = model_data['model'].predict(features_array)[0]
            
            # Get class name
            predicted_class = model_data['class_names'][prediction]
            
            # Get prediction probabilities
            probabilities = model_data['model'].predict_proba(features_array)[0]
            class_probs = {
                model_data['class_names'][i]: float(prob)
                for i, prob in enumerate(probabilities)
            }
            
            return jsonify({
                "predicted_class": predicted_class,
                "predicted_class_index": int(prediction),
                "probabilities": class_probs
            })
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    else:
        return jsonify({"error": "Unsupported HTTP method"}), 405


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model_loaded": model_loaded
    })


if __name__ == '__main__':
    # Load model on startup
    print("Starting Fruit Classification API...")
    load_model()
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)

