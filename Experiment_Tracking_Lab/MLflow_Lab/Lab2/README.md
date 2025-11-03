# Customer Churn Prediction Lab Documentation

This documentation provides a step-by-step guide to a complete MLOps pipeline focused on predicting customer churn using Python, MLflow, and various machine learning libraries. The lab covers data preprocessing, model training, model registration, batch inference, and real-time deployment.

## Prerequisites

Before starting the lab, ensure that you have the following:

- Python environment set up with required libraries installed (see `requirements.txt`)
- **Dataset**: You will need the Telco Customer Churn CSV file: `WA_Fn-UseC_-Telco-Customer-Churn.csv`
  - Download from: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
  - Place the file in the `Lab2/data/` directory

## Step 1: Importing Data

In this step, we load the Telco Customer Churn dataset using the Pandas library.

```python
import pandas as pd

# Load the Telco Customer Churn dataset
data_path = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(data_path)
```

### Explanation:
- We use Pandas to read the CSV file containing customer churn data
- The dataset contains customer information and whether they churned (Yes/No)
- Expected dataset shape: approximately 7,043 rows and 21 columns

## Step 2: Exploring Data

In this step, we'll explore the customer churn dataset by examining the data structure, columns, and basic statistics.

### Code:

```python
# Display first few rows
df.head()

# Display dataset information
df.info()

# Display statistical summary
df.describe()
```

### Explanation:
- `head()` displays the first 5 rows of the dataset
- `info()` shows data types and memory usage
- `describe()` provides statistical summary of numeric columns
- This helps understand the data structure and identify potential issues

## Step 3: Data Preprocessing

In this step, we'll perform data preprocessing tasks to prepare the customer churn dataset for model training.

### Handling Missing Values

The `TotalCharges` column may contain empty strings that need to be converted to numeric values.

### Code:

```python
import numpy as np

# Check for missing values
print("Missing values before preprocessing:")
print(df.isnull().sum())

# Convert TotalCharges to numeric, handling empty strings
df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Fill missing values with median
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
```

### Converting Target Variable

Convert the Churn column from "Yes"/"No" to binary (1/0).

```python
# Remove spaces from column names
df.rename(columns=lambda x: x.replace(' ', '_'), inplace=True)

# Convert Churn to binary (Yes=1, No=0)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
```

### Explanation:
- Empty strings in `TotalCharges` are converted to NaN
- Missing values are filled with the median value
- The target variable is converted from categorical to binary for classification
- Column names are cleaned (spaces replaced with underscores)

## Step 4: Data Visualization

In this step, we'll visualize the distribution of the 'Churn' target variable in the dataset.

### Code:

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Visualize the distribution of the Churn target variable
plt.figure(figsize=(8, 5))
sns.countplot(x='Churn', data=df)
plt.title('Customer Churn Distribution')
plt.xlabel('Churn (0=No, 1=Yes)')
plt.ylabel('Count')
plt.show()
```

### Explanation:
- Creates a bar plot showing the distribution of churn vs. no churn
- Helps understand class imbalance (typically ~73% No Churn, ~27% Churn)
- Expected Output: Bar chart showing the count of customers who churned vs. did not churn

## Step 5: Feature Engineering

In this step, we'll create new features to improve model performance.

### Creating Tenure Groups

```python
# Create tenure groups (new, mid, long-term customers)
def categorize_tenure(tenure):
    if tenure <= 24:
        return 'New'
    elif tenure <= 48:
        return 'Mid'
    else:
        return 'Long-term'

df['tenure_group'] = df['tenure'].apply(categorize_tenure)
```

### Creating Monthly Charges Ratio

```python
# Monthly charges per tenure ratio
df['monthly_charges_per_tenure'] = df['MonthlyCharges'] / (df['tenure'] + 1)
```

### Explanation:
- **Tenure Groups**: Categorizes customers into new (≤24 months), mid (25-48 months), and long-term (>48 months)
- **Monthly Charges Ratio**: Creates a ratio feature that normalizes monthly charges by tenure
- These engineered features often improve model performance by capturing domain knowledge

## Step 6: Encode Categorical Variables and Prepare Data

In this step, we'll encode categorical variables and prepare the data for training.

### Code:

```python
from sklearn.preprocessing import LabelEncoder

# Separate features and target
X = df.drop(['customerID', 'Churn'], axis=1)
y = df['Churn']

# Encode categorical variables using label encoding
le_dict = {}
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    le_dict[col] = le
```

### Explanation:
- Drops customerID (identifier, not a feature) and Churn (target variable) from features
- Encodes categorical variables using LabelEncoder
- Saves encoders in a dictionary for later use (if needed for inference)
- Expected Output: Numeric features ready for machine learning

## Step 7: Exploratory Data Analysis (EDA)

In this step, we'll perform Exploratory Data Analysis by creating box plots to identify potential predictors of customer churn.

### Code:

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Select numeric columns for visualization
numeric_cols = X.select_dtypes(include=[np.number]).columns
plot_cols = numeric_cols[:12]  # Limit to 12 features

# Create subplots
dims = (3, 4)
f, axes = plt.subplots(dims[0], dims[1], figsize=(25, 15))
axes = axes.flatten()

for idx, col in enumerate(plot_cols):
    sns.boxplot(x=y, y=X[col], ax=axes[idx])
    axes[idx].set_title(f'{col} by Churn')
    axes[idx].set_xlabel('Churn (0=No, 1=Yes)')

plt.tight_layout()
plt.show()
```

### Explanation:
- Creates box plots comparing feature distributions for churned vs. non-churned customers
- Helps identify features that differ significantly between classes
- Features with clear separation between churn/no-churn are likely good predictors
- Expected Output: Grid of box plots showing feature distributions by churn status

## Step 8: Data Splitting

In this step, we'll split the dataset into training, validation, and test sets to prepare for model training and evaluation.

### Code:

```python
from sklearn.model_selection import train_test_split

X = df.drop(['customerID', 'Churn'], axis=1)
y = df['Churn']

# Split out the training data (60%)
X_train, X_rem, y_train, y_rem = train_test_split(
    X, y, train_size=0.6, random_state=123, stratify=y
)

# Split the remaining data equally into validation and test (20% each)
X_val, X_test, y_val, y_test = train_test_split(
    X_rem, y_rem, test_size=0.5, random_state=123, stratify=y_rem
)
```

### Explanation:
- **Train set (60%)**: Used to train the model
- **Validation set (20%)**: Used for hyperparameter tuning (optional)
- **Test set (20%)**: Used for final model evaluation
- `stratify=y` ensures class distribution is maintained across splits
- `random_state=123` ensures reproducibility

## Step 9: Building a Baseline Model

In this step, we'll create a baseline model using a Random Forest classifier and log its performance using MLflow.

### Creating Model Wrapper

```python
import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from mlflow.models.signature import infer_signature
from mlflow.utils.environment import _mlflow_conda_env
import cloudpickle
import time

# The predict method of sklearn's RandomForestClassifier returns binary classification (0 or 1).
# The following code creates a wrapper function, SklearnModelWrapper, that uses 
# the predict_proba method to return the probability that the observation belongs to the positive class.

class SklearnModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model
    
    def predict(self, context, model_input):
        # Return probability of positive class (churn=1)
        return self.model.predict_proba(model_input)[:, 1]
```

### Training and Logging Model

```python
# mlflow.start_run creates a new MLflow run to track the performance of this model. 
# Within the context, you call mlflow.log_param to keep track of the parameters used, and
# mlflow.log_metric to record metrics like accuracy.

with mlflow.start_run(run_name='churn_baseline_model'):
    n_estimators = 100
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=np.random.RandomState(123)
    )
    model.fit(X_train, y_train)
    
    # predict_proba returns [prob_negative, prob_positive], so slice the output with [:, 1]
    predictions_test = model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, predictions_test)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    precision = precision_score(y_test, model.predict(X_test))
    recall = recall_score(y_test, model.predict(X_test))
    f1 = f1_score(y_test, model.predict(X_test))
    
    # Log hyperparameters
    mlflow.log_param('n_estimators', n_estimators)
    
    # Log metrics
    mlflow.log_metric('auc', auc_score)
    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_metric('precision', precision)
    mlflow.log_metric('recall', recall)
    mlflow.log_metric('f1_score', f1)
    
    # Create wrapped model
    wrappedModel = SklearnModelWrapper(model)
    
    # Log the model with a signature that defines the schema of the model's inputs and outputs. 
    # When the model is deployed, this signature will be used to validate inputs.
    signature = infer_signature(X_train, wrappedModel.predict(None, X_train))
    
    # MLflow contains utilities to create a conda environment used to serve models.
    # The necessary dependencies are added to a conda.yaml file which is logged along with the model.
    conda_env = _mlflow_conda_env(
        additional_conda_deps=None,
        additional_pip_deps=[
            f"cloudpickle=={cloudpickle.__version__}",
            f"scikit-learn=={sklearn.__version__}"
        ],
        additional_conda_channels=None,
    )
    
    mlflow.pyfunc.log_model(
        "random_forest_model",
        python_model=wrappedModel,
        conda_env=conda_env,
        signature=signature
    )
```

### Explanation:
- We create a Random Forest classifier model using scikit-learn's RandomForestClassifier
- The model is trained on the training data (X_train, y_train)
- We log various information using MLflow, including model parameters (n_estimators), metrics (AUC, accuracy, precision, recall, F1), and the model itself
- A wrapper class SklearnModelWrapper is used to make predictions using predict_proba, which returns class probabilities
- We also define a signature to validate inputs when the model is deployed
- Expected Output: Model training details and metrics (e.g., AUC ~0.85) will be logged in the MLflow run. The trained model will be saved for future use.

## Step 10: Feature Importance Analysis

In this step, we analyze feature importance to identify which features have the most impact on predicting customer churn.

### Code:

```python
feature_importances = pd.DataFrame(
    model.feature_importances_,
    index=X_train.columns.tolist(),
    columns=['importance']
)
feature_importances = feature_importances.sort_values('importance', ascending=False)

print("Top 10 Most Important Features:")
print(feature_importances.head(10))
```

### Explanation:
- We calculate the feature importances using the trained random forest classifier model
- The `model.feature_importances_` attribute provides the importance scores for each feature
- We create a DataFrame to display the importances along with feature names
- Finally, we sort the DataFrame in descending order to identify the most important features
- Expected Output: A table showing the feature importances in descending order, with the most important features at the top

## Step 11: Model Registration in MLflow Model Registry

In this step, we'll register the trained model in the MLflow Model Registry for version tracking and management.

### Code:

```python
# Search for the run using run name
runs = mlflow.search_runs(filter_string='tags.mlflow.runName = "churn_baseline_model"')
run_id = runs.iloc[0].run_id

model_name = "customer_churn"
model_version = mlflow.register_model(
    f"runs:/{run_id}/random_forest_model",
    model_name
)

# Registering the model takes a few seconds, so add a small delay
time.sleep(15)
```

### Explanation:
- We retrieve the run ID of the MLflow run where the model was trained using `mlflow.search_runs`
- We specify the desired model name, in this case, "customer_churn"
- We use `mlflow.register_model` to register the model in the Model Registry. The path to the model is constructed using the run ID
- A delay is added to ensure the model registration process is completed
- Expected Output: The trained model will be registered in the MLflow Model Registry under the specified model name ("customer_churn")

### Note:
Model registration in the MLflow Model Registry allows for versioning and tracking of different model versions. It's a crucial step for managing and deploying machine learning models in a production environment.

## Step 12: Transitioning Model Version to Production

In this step, we'll transition the newly registered model version to the "Production" stage in the MLflow Model Registry.

### Code:

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()
client.transition_model_version_stage(
    name=model_name,
    version=model_version.version,
    stage="Production"
)
```

### Explanation:
- We use the MlflowClient to interact with the MLflow Tracking Server programmatically
- The `client.transition_model_version_stage` method is used to transition the model version to the "Production" stage
- Expected Output: The model version will be moved to the "Production" stage in the MLflow Model Registry

### Note:
Transitioning a model version to "Production" indicates that it is ready for use in a production environment. You can now refer to the model using the path `models:/customer_churn/production`. This step is crucial for managing the deployment of machine learning models.

## Step 13: Model Inference and Evaluation

In this step, we'll load the production version of the model from the MLflow Model Registry and perform batch inference and evaluation.

### Code:

```python
from sklearn.metrics import roc_auc_score, accuracy_score

# Load the production model
production_model = mlflow.pyfunc.load_model(f"models:/{model_name}/production")

# Make predictions on test set
test_predictions = production_model.predict(X_test)

# Calculate metrics
test_auc = roc_auc_score(y_test, test_predictions)
test_accuracy = accuracy_score(y_test, (test_predictions > 0.5).astype(int))

print(f'AUC: {test_auc:.4f}')
print(f'Accuracy: {test_accuracy:.4f}')
```

### Explanation:
- We load the production version of the model from the MLflow Model Registry using `mlflow.pyfunc.load_model`
- We perform batch inference on the test data (X_test) using the loaded model
- We calculate and print the Area Under the ROC Curve (AUC) score and accuracy to assess the model's performance
- Expected Output: The AUC score and accuracy will be printed, providing an evaluation of the model's performance on the test data

### Note:
Loading the production model version allows us to make predictions on new data. The AUC score is used here as an example metric for model evaluation. Depending on the problem, other evaluation metrics may be more appropriate.

## Step 14: Batch Inference with the Deployed Model

In this step, we'll perform batch inference using the deployed model to make predictions on a dataset.

### Code:

```python
import mlflow.pyfunc

# Load the production model
batch_model = mlflow.pyfunc.load_model(f"models:/{model_name}/production")

# Perform batch predictions on test dataset
batch_predictions = batch_model.predict(X_test)

# Add predictions to results
test_results = X_test.copy()
test_results['actual_churn'] = y_test.values
test_results['predicted_churn_prob'] = batch_predictions
test_results['predicted_churn'] = (batch_predictions > 0.5).astype(int)

print("Batch Inference Results:")
print(f"Total predictions: {len(batch_predictions)}")
print(f"Average churn probability: {batch_predictions.mean():.4f}")
test_results[['actual_churn', 'predicted_churn_prob', 'predicted_churn']].head(10)
```

### Explanation:
- We load the production model from the Model Registry
- We apply the model to the test dataset to make batch predictions
- We create a results DataFrame that includes actual values, predicted probabilities, and binary predictions
- Expected Output: The predictions DataFrame will contain the input data along with predictions

### Note:
Batch inference is the process of using a trained machine learning model to make predictions on a batch of data. In this step, we apply the deployed model to the test dataset to make predictions at scale.

## Step 15: Serving the Model for Real-Time Inference

In this step, we'll serve the model to enable real-time inference using MLflow's model serving capabilities.

### Code (Command Line):

```bash
# Serve the model using the MLflow Model Serving
mlflow models serve -m models:/customer_churn/production -h 0.0.0.0 -p 5001
```

### Explanation:
- We use the `mlflow models serve` command to start serving the model
- `-m models:/customer_churn/production` specifies the model to be served
- `-h 0.0.0.0` specifies the host on which the model will be served
- `-p 5001` specifies the port on which the model will be available for real-time inference

### Expected Output:
The model will be served and ready to accept real-time inference requests.

### Note:
Serving a model allows you to make real-time predictions by sending requests to the model's API endpoint. Ensure that the MLflow Model Server is correctly configured and running before serving the model. Real-time serving is useful for integrating machine learning models into production systems, applications, or APIs.

## Step 16: Performing Real-Time Inference with the Deployed Model

In this step, we'll perform real-time inference by sending requests to the deployed model's API endpoint.

### Code (Python):

```python
import requests
import json
import time

# Wait for the server to start
time.sleep(5)

# Prepare the data for prediction
test_sample = X_test.head(5)

# Format data for MLflow API (dataframe_split format)
data_dict = {"dataframe_split": test_sample.to_dict(orient='split')}

# Send POST request to the model server
url = 'http://localhost:5001/invocations'

response = requests.post(url, json=data_dict)
predictions = response.json()

print("Real-time inference predictions:")
print(predictions)
```

### Explanation:
- We use the `requests` library to send a POST request to the deployed model's API endpoint
- The URL should be set to the correct endpoint where the model is served (`http://localhost:5001/invocations`)
- We prepare the input data in the desired format (dataframe_split) and send it as JSON in the request
- The response contains the model's predictions, which we extract using `response.json()`
- Finally, we print the predictions

### Expected Output:
The output will be the model's predictions for the input data sent in the request.

### Note:
Real-time inference allows you to use the deployed model to make predictions on new data as it becomes available. Ensure that the model serving endpoint is running and accessible before making real-time inference requests. Replace the endpoint URL with the actual URL where your model is served.

## Step 17: Cleaning Up and Conclusion

In this final step, we'll wrap up the lab and perform any necessary clean-up tasks.

### Clean-Up Tasks:

- **Stop the Model Serving**: If you've started the model serving process, make sure to stop it when you're done with real-time inference. You can do this by stopping the MLflow Model Serving process or using appropriate commands.
- **Close Resources**: Ensure that any resources or connections used during the lab are properly closed or released.
- **Save Documentation**: Save this lab documentation for future reference or sharing with others.

### Conclusion:

In this lab, we've covered various aspects of the machine learning lifecycle, including data preparation, model training, evaluation, deployment, and real-time inference. Here are the key takeaways:

- **Data preparation** is essential for training and evaluating machine learning models. Cleaning, transforming, and splitting the data are crucial steps.
- **Model training** involves selecting an appropriate algorithm, training the model, and evaluating its performance using relevant metrics.
- **Model deployment** involves registering the model, transitioning it to the production stage, and serving it for real-time inference.
- **Real-time inference** allows you to use the deployed model to make predictions on new data as it arrives.

By following these steps, you can effectively develop and deploy machine learning models for various applications.

Thank you for completing this lab!

