"""
Wine Quality Classification Pipeline - Source Module

Author: Mohan Bhosale
Date: October 2025
Course: MLOps - Fall 2025

This module contains the core machine learning functions for the wine quality
classification pipeline. These functions are orchestrated by Apache Airflow.

Functions:
- load_wine_data: Loads and explores the Red Wine Quality dataset
- perform_feature_engineering: Creates features and preprocesses data
- train_random_forest_model: Trains a Random Forest classifier
- evaluate_model_performance: Evaluates model and generates reports
"""

__version__ = "1.0.0"
__author__ = "Mohan Bhosale"

