"""
Wine Quality Classification Pipeline
Author: Mohan Bhosale
Description: ML pipeline for predicting wine quality using Random Forest classifier

This module contains the core functions for the wine quality classification pipeline.
Each function represents a step in the ML workflow that will be orchestrated by Airflow.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import os
import json


def load_wine_data():
    """
    Step 1: Load and explore the wine quality dataset
    
    This function reads the Red Wine Quality dataset from Kaggle and performs
    initial data exploration to understand the structure and characteristics.
    
    Returns:
        bytes: Serialized dictionary containing the dataframe and basic statistics
    """
    print("=" * 60)
    print("TASK 1: Loading Red Wine Quality Dataset")
    print("=" * 60)
    
    # Construct the path to our dataset
    # The dataset is stored in the data folder relative to this script
    data_path = os.path.join(os.path.dirname(__file__), "../data/Red Wine Quality.csv")
    
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(data_path)
    
    # Let's explore the dataset to understand what we're working with
    print(f"\n📊 Dataset Overview:")
    print(f"   - Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"   - Features: {list(df.columns)}")
    
    # Check for any missing data that might need handling
    missing_count = df.isnull().sum().sum()
    print(f"\n🔍 Data Quality Check:")
    print(f"   - Missing values: {missing_count}")
    if missing_count == 0:
        print("   - ✓ No missing data - dataset is clean!")
    
    # Understand the distribution of wine quality ratings
    print(f"\n🍷 Wine Quality Distribution:")
    quality_counts = df['quality'].value_counts().sort_index()
    for quality, count in quality_counts.items():
        print(f"   - Quality {quality}: {count} wines")
    
    # Show a preview of the data
    print(f"\n📋 Sample Data (first 3 rows):")
    print(df.head(3).to_string(index=False))
    
    # Package up statistics that we'll need later
    stats = {
        'total_samples': len(df),
        'num_features': df.shape[1],
        'missing_values': missing_count,
        'quality_distribution': quality_counts.to_dict()
    }
    
    # Combine dataframe and statistics into a single package
    # We'll serialize this to pass between Airflow tasks
    result = {
        'dataframe': df,
        'statistics': stats
    }
    
    serialized_data = pickle.dumps(result)
    print(f"\n✅ Data loaded successfully! Ready for preprocessing.")
    print("=" * 60)
    
    return serialized_data


def perform_feature_engineering(data):
    """
    Step 2: Engineer features and preprocess the data
    
    This function transforms the raw wine data into a format suitable for machine learning.
    We'll create new features, handle the target variable, and prepare train/test splits.
    
    Args:
        data (bytes): Serialized dataset from the load_wine_data task
        
    Returns:
        bytes: Serialized dictionary containing processed train/test data and scaler
    """
    print("=" * 60)
    print("TASK 2: Feature Engineering & Data Preprocessing")
    print("=" * 60)
    
    # Unpack the data from the previous task
    result = pickle.loads(data)
    df = result['dataframe']
    
    # The original dataset has quality ratings from 3-8
    # Let's simplify this to a binary classification problem:
    # - "Good wine" = quality rating of 6 or higher
    # - "Bad wine" = quality rating below 6
    df['is_good_wine'] = (df['quality'] >= 6).astype(int)
    
    good_count = df['is_good_wine'].sum()
    bad_count = len(df) - good_count
    
    print(f"\n🎯 Target Variable Created:")
    print(f"   - Good wines (quality ≥ 6): {good_count} samples")
    print(f"   - Bad wines (quality < 6): {bad_count} samples")
    print(f"   - Class balance: {good_count/len(df)*100:.1f}% good vs {bad_count/len(df)*100:.1f}% bad")
    
    # Now let's create some domain-specific features
    # These new features might help the model learn better patterns
    print(f"\n🔧 Engineering New Features:")
    
    # Feature 1: Alcohol relative to density (higher might indicate quality)
    df['alcohol_to_density'] = df['alcohol'] / df['density']
    print(f"   1. alcohol_to_density = alcohol / density")
    
    # Feature 2: Ratio of fixed to volatile acidity (balance might matter)
    df['acid_ratio'] = df['fixed acidity'] / df['volatile acidity']
    print(f"   2. acid_ratio = fixed acidity / volatile acidity")
    
    # Feature 3: Ratio of free to total sulfur dioxide (preservation metric)
    df['sulfur_ratio'] = df['free sulfur dioxide'] / df['total sulfur dioxide']
    print(f"   3. sulfur_ratio = free sulfur dioxide / total sulfur dioxide")
    
    # Clean up any division-by-zero issues (replace inf with NaN, then fill with median)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(df.median())
    
    # Separate features from target variable
    # Drop both the original 'quality' and our new 'is_good_wine' target
    X = df.drop(['quality', 'is_good_wine'], axis=1)
    y = df['is_good_wine']
    
    print(f"\n📦 Final Feature Set: {X.shape[1]} features")
    print(f"   {list(X.columns)}")
    
    # Split into training and test sets (80/20 split)
    # We use stratify to maintain class balance in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2,      # Hold out 20% for testing
        random_state=42,     # For reproducibility
        stratify=y          # Keep class distribution balanced
    )
    
    print(f"\n✂️ Data Split Complete:")
    print(f"   - Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(df)*100:.1f}%)")
    print(f"   - Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(df)*100:.1f}%)")
    
    # Feature scaling is crucial for many ML algorithms
    # StandardScaler transforms features to have mean=0 and std=1
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Fit on training data only!
    X_test_scaled = scaler.transform(X_test)        # Apply same transformation to test
    
    print(f"\n⚖️ Feature Scaling Applied:")
    print(f"   - Method: StandardScaler (mean=0, std=1)")
    print(f"   - Note: Scaler fitted on training data only to prevent data leakage")
    
    # Package everything up for the next task
    processed_data = {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': list(X.columns),
        'scaler': scaler
    }
    
    serialized_result = pickle.dumps(processed_data)
    print(f"\n✅ Feature engineering completed successfully!")
    print("=" * 60)
    
    return serialized_result


def train_random_forest_model(data, model_filename):
    """
    Step 3: Train a Random Forest classifier
    
    Random Forest is an ensemble learning method that builds multiple decision trees
    and combines their predictions. It's robust, handles non-linear relationships well,
    and provides feature importance scores.
    
    Args:
        data (bytes): Serialized preprocessed data from feature engineering task
        model_filename (str): Filename to save the trained model
        
    Returns:
        dict: Training results including cross-validation scores and feature importances
    """
    print("=" * 60)
    print("TASK 3: Training Random Forest Classifier")
    print("=" * 60)
    
    # Unpack the preprocessed data
    processed_data = pickle.loads(data)
    X_train = processed_data['X_train']
    y_train = processed_data['y_train']
    feature_names = processed_data['feature_names']
    
    # Configure our Random Forest model
    # These hyperparameters were chosen based on common best practices:
    # - n_estimators: More trees = better performance (but slower)
    # - max_depth: Prevents overfitting by limiting tree depth
    # - min_samples_split: Minimum samples required to split a node
    # - n_jobs: Use all CPU cores for parallel processing
    print(f"\n🌲 Random Forest Configuration:")
    print(f"   - n_estimators: 100 (number of trees in the forest)")
    print(f"   - max_depth: 10 (maximum depth of each tree)")
    print(f"   - min_samples_split: 5 (min samples to split a node)")
    print(f"   - random_state: 42 (for reproducibility)")
    print(f"   - n_jobs: -1 (use all CPU cores)")
    
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1  # Use all available CPU cores
    )
    
    # Before training on all data, let's validate with cross-validation
    # This helps us understand how well the model generalizes
    print(f"\n🔄 Running 5-Fold Cross-Validation...")
    print(f"   (This splits training data into 5 parts, trains on 4, validates on 1, repeats 5 times)")
    
    cv_scores = cross_val_score(
        rf_model, 
        X_train, 
        y_train, 
        cv=5,              # 5-fold cross-validation
        scoring='accuracy'  # Measure accuracy
    )
    
    print(f"\n📊 Cross-Validation Results:")
    print(f"   - Fold 1: {cv_scores[0]:.4f}")
    print(f"   - Fold 2: {cv_scores[1]:.4f}")
    print(f"   - Fold 3: {cv_scores[2]:.4f}")
    print(f"   - Fold 4: {cv_scores[3]:.4f}")
    print(f"   - Fold 5: {cv_scores[4]:.4f}")
    print(f"   - Mean Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Now train the final model on the full training set
    print(f"\n🎯 Training final model on full training set...")
    rf_model.fit(X_train, y_train)
    print(f"   ✅ Training complete!")
    
    # One of the great features of Random Forest is feature importance
    # This tells us which features the model thinks are most predictive
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n🔍 Feature Importance Analysis:")
    print(f"   (Which features does the model think matter most?)\n")
    for idx, row in feature_importance.head(5).iterrows():
        # Create a visual bar using █ characters
        bar = '█' * int(row['importance'] * 100)
        print(f"   {idx+1}. {row['feature']:<25} {row['importance']:.4f} {bar}")
    
    # Save the trained model to disk
    # We'll need this in the next task for evaluation
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, model_filename)
    with open(model_path, 'wb') as f:
        pickle.dump(rf_model, f)
    
    print(f"\n💾 Model Artifacts Saved:")
    print(f"   - Model: {model_path}")
    
    # Also save the scaler - we'll need it if we want to make predictions later
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(processed_data['scaler'], f)
    
    print(f"   - Scaler: {scaler_path}")
    
    # Package up the results to pass to the next task
    training_results = {
        'cv_scores': cv_scores.tolist(),
        'mean_cv_score': float(cv_scores.mean()),
        'std_cv_score': float(cv_scores.std()),
        'feature_importance': feature_importance.to_dict('records')
    }
    
    print(f"\n✅ Model training completed successfully!")
    print("=" * 60)
    
    return training_results


def evaluate_model_performance(model_filename, training_results, **context):
    """
    Step 4: Evaluate the model and generate performance report
    
    This final step loads our trained model, runs it on the held-out test set,
    and calculates comprehensive performance metrics. The results are saved to
    a JSON file for future reference.
    
    Args:
        model_filename (str): Name of the saved model file
        training_results (dict): Training metrics from the previous task
        context: Airflow context (used to pull data from previous tasks via XCom)
        
    Returns:
        dict: Complete evaluation metrics including accuracy, precision, recall, etc.
    """
    print("=" * 60)
    print("TASK 4: Model Evaluation & Performance Analysis")
    print("=" * 60)
    
    # Load the trained model from disk
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
    model_path = os.path.join(model_dir, model_filename)
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    print(f"\n📂 Loaded trained model from: {model_path}")
    
    # Retrieve the test data from the feature engineering task
    # XCom allows us to pass data between Airflow tasks
    ti = context['ti']
    data = ti.xcom_pull(task_ids='feature_engineering_task')
    processed_data = pickle.loads(data)
    
    X_test = processed_data['X_test']
    y_test = processed_data['y_test']
    
    print(f"📊 Running evaluation on {len(X_test)} test samples...")
    
    # Generate predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)  # Probability scores
    
    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    # Display results in a clear, organized format
    print("\n" + "=" * 60)
    print("📈 EVALUATION RESULTS")
    print("=" * 60)
    
    print(f"\n🎯 Overall Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Classification report shows precision, recall, and F1 for each class
    print(f"\n📋 Detailed Classification Metrics:\n")
    print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support'}")
    print("-" * 68)
    print(f"{'Bad Wine (0)':<20} {class_report['0']['precision']:<12.4f} {class_report['0']['recall']:<12.4f} {class_report['0']['f1-score']:<12.4f} {int(class_report['0']['support'])}")
    print(f"{'Good Wine (1)':<20} {class_report['1']['precision']:<12.4f} {class_report['1']['recall']:<12.4f} {class_report['1']['f1-score']:<12.4f} {int(class_report['1']['support'])}")
    
    # Confusion matrix shows where the model makes mistakes
    print(f"\n🔢 Confusion Matrix:")
    print(f"   (Rows = Actual, Columns = Predicted)\n")
    print(f"                    Predicted Bad    Predicted Good")
    print(f"   Actual Bad         {conf_matrix[0][0]:<16} {conf_matrix[0][1]:<16}")
    print(f"   Actual Good        {conf_matrix[1][0]:<16} {conf_matrix[1][1]:<16}")
    
    # Calculate additional useful metrics
    tn, fp, fn, tp = conf_matrix.ravel()
    
    # Sensitivity (True Positive Rate): Of all actual good wines, how many did we catch?
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # Specificity (True Negative Rate): Of all actual bad wines, how many did we correctly identify?
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"\n📊 Additional Performance Metrics:")
    print(f"   - Sensitivity (True Positive Rate):  {sensitivity:.4f}")
    print(f"   - Specificity (True Negative Rate):  {specificity:.4f}")
    print(f"\n   Breakdown:")
    print(f"   - True Positives (correctly identified good wines):   {tp}")
    print(f"   - True Negatives (correctly identified bad wines):    {tn}")
    print(f"   - False Positives (bad wines wrongly called good):    {fp}")
    print(f"   - False Negatives (good wines wrongly called bad):    {fn}")
    
    # Compare with training performance
    print(f"\n🔄 Training vs Test Performance:")
    print(f"   - Cross-Validation Score: {training_results['mean_cv_score']:.4f} ± {training_results['std_cv_score']:.4f}")
    print(f"   - Test Set Accuracy:      {accuracy:.4f}")
    
    # Check for overfitting
    performance_gap = abs(training_results['mean_cv_score'] - accuracy)
    if performance_gap < 0.05:
        print(f"   - ✅ Model generalizes well (gap < 5%)")
    else:
        print(f"   - ⚠️  Performance gap is {performance_gap:.2%} - may indicate overfitting")
    
    # Compile all metrics into a report
    evaluation_report = {
        'test_accuracy': float(accuracy),
        'classification_report': class_report,
        'confusion_matrix': conf_matrix.tolist(),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'training_cv_score': training_results['mean_cv_score'],
        'top_features': training_results['feature_importance'][:5]
    }
    
    # Save the report as a JSON file for documentation
    report_path = os.path.join(model_dir, "evaluation_report.json")
    with open(report_path, 'w') as f:
        json.dump(evaluation_report, f, indent=2)
    
    print(f"\n💾 Evaluation report saved to: {report_path}")
    
    # Final summary
    print("\n" + "=" * 60)
    print("🎉 WINE QUALITY PREDICTION PIPELINE COMPLETED!")
    print("=" * 60)
    
    # Provide a human-readable assessment
    if accuracy > 0.85:
        quality_rating = "EXCELLENT"
        emoji = "🌟"
    elif accuracy > 0.75:
        quality_rating = "GOOD"
        emoji = "👍"
    elif accuracy > 0.65:
        quality_rating = "FAIR"
        emoji = "👌"
    else:
        quality_rating = "NEEDS IMPROVEMENT"
        emoji = "⚠️"
    
    print(f"\n{emoji} Final Model Performance: {accuracy*100:.2f}% accuracy")
    print(f"{emoji} Model Quality Rating: {quality_rating}")
    print(f"\n💡 This model can predict wine quality with ~{accuracy*100:.1f}% accuracy")
    print(f"   based on physicochemical properties of the wine.")
    
    print("\n" + "=" * 60 + "\n")
    
    return evaluation_report

