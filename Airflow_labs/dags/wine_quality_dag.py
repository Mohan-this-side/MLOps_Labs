"""
Wine Quality Classification DAG
Author: Mohan Bhosale
Date: October 2025

This Airflow DAG orchestrates an end-to-end machine learning pipeline for 
predicting wine quality based on physicochemical properties.

Pipeline Steps:
1. Load and explore the Red Wine Quality dataset from Kaggle
2. Engineer new features and preprocess the data (scaling, train/test split)
3. Train a Random Forest classifier with cross-validation
4. Evaluate model performance and generate a comprehensive report

The pipeline demonstrates MLOps best practices including:
- Workflow orchestration with Airflow
- Data passing between tasks using XCom
- Model versioning and artifact storage
- Automated performance evaluation
"""

from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta

# Import our custom ML pipeline functions
from src.wine_pipeline import (
    load_wine_data,
    perform_feature_engineering,
    train_random_forest_model,
    evaluate_model_performance
)
from airflow import configuration as conf

# Enable pickle support for XCom
# This allows us to pass complex Python objects (like DataFrames and arrays) between tasks
# Without this, XCom would only support simple types like strings and numbers
conf.set('core', 'enable_xcom_pickling', 'True')

# Define default arguments that apply to all tasks in this DAG
default_args = {
    'owner': 'Mohan Bhosale',              # Owner of this DAG (for tracking purposes)
    'start_date': datetime(2025, 10, 6),   # When this DAG becomes active
    'retries': 1,                          # If a task fails, retry it once
    'retry_delay': timedelta(minutes=3),   # Wait 3 minutes before retrying
    'email_on_failure': False,             # Don't send email on failure (set True in production)
    'email_on_retry': False,               # Don't send email on retry
}

# Create the DAG instance
# This is the main container for our workflow
dag = DAG(
    dag_id='Wine_Quality_ML_Pipeline',        # Unique identifier for this DAG
    default_args=default_args,                # Apply the default arguments defined above
    description='End-to-end ML pipeline for wine quality classification using Random Forest',
    schedule_interval=None,                   # None = manual trigger only
                                              # Could also use: '@daily', '@weekly', '0 9 * * *' (cron)
    catchup=False,                            # Don't run for past dates when DAG is turned on
    tags=['machine-learning', 'classification', 'wine-quality', 'random-forest'],  # Tags for filtering in UI
)

# ============================================================================
# TASK DEFINITIONS
# ============================================================================
# Each task is a step in our ML pipeline. Tasks are defined using PythonOperator
# which executes a Python function. Data flows between tasks via XCom.

# Task 1: Load Wine Quality Dataset
# This task reads the CSV file and performs initial data exploration
load_data_task = PythonOperator(
    task_id='load_wine_data_task',        # Unique ID for this task
    python_callable=load_wine_data,        # The function to execute
    dag=dag,                               # Associate with our DAG
)

# Task 2: Feature Engineering & Preprocessing
# This task creates new features, splits data, and scales features
feature_engineering_task = PythonOperator(
    task_id='feature_engineering_task',
    python_callable=perform_feature_engineering,
    op_args=[load_data_task.output],      # Pass the output from Task 1 as input
    dag=dag,
)

# Task 3: Train Random Forest Model
# This task trains the model with cross-validation and saves it to disk
train_model_task = PythonOperator(
    task_id='train_model_task',
    python_callable=train_random_forest_model,
    op_args=[
        feature_engineering_task.output,   # Preprocessed data from Task 2
        "wine_quality_model.pkl"           # Filename to save the model
    ],
    dag=dag,
)

# Task 4: Evaluate Model Performance
# This task loads the model, evaluates it on test data, and generates a report
evaluate_model_task = PythonOperator(
    task_id='evaluate_model_task',
    python_callable=evaluate_model_performance,
    op_args=[
        "wine_quality_model.pkl",          # Model filename to load
        train_model_task.output            # Training metrics from Task 3
    ],
    provide_context=True,                  # Enables access to Airflow context (needed for XCom pull)
    dag=dag,
)

# ============================================================================
# TASK DEPENDENCIES (DAG Structure)
# ============================================================================
# The >> operator defines task dependencies (what runs after what)
# This creates a linear pipeline: Task1 → Task2 → Task3 → Task4

load_data_task >> feature_engineering_task >> train_model_task >> evaluate_model_task

# Visual representation of the pipeline:
# 
#  ┌─────────────────┐
#  │  Load Data      │  ← Read CSV, explore dataset
#  └────────┬────────┘
#           │
#           ▼
#  ┌─────────────────┐
#  │  Feature Eng.   │  ← Create features, split, scale
#  └────────┬────────┘
#           │
#           ▼
#  ┌─────────────────┐
#  │  Train Model    │  ← Train Random Forest with CV
#  └────────┬────────┘
#           │
#           ▼
#  ┌─────────────────┐
#  │  Evaluate       │  ← Test model, generate report
#  └─────────────────┘

# Alternative syntax for defining dependencies (same result):
# load_data_task.set_downstream(feature_engineering_task)
# feature_engineering_task.set_downstream(train_model_task)
# train_model_task.set_downstream(evaluate_model_task)

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

if __name__ == "__main__":
    # This allows you to interact with the DAG from the command line
    # For example: python wine_quality_dag.py test load_wine_data_task 2025-10-06
    dag.cli()

