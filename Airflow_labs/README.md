# Wine Quality Classification with Apache Airflow

**Author:** Mohan Bhosale  
**Date:** October 2025  
**Course:** MLOps - Fall 2025

---

## 🎉 Pipeline Success - Results Achieved!

**✅ Successfully deployed and executed ML pipeline with 80.6% accuracy!**

The Wine Quality Classification pipeline has been successfully implemented using Apache Airflow and Docker. All tasks completed successfully, generating trained models and comprehensive evaluation reports.

![Successful Pipeline Execution](<img width="1869" height="665" alt="image" src="https://github.com/user-attachments/assets/e8e5375e-89ba-4da5-b9f4-50d7eae64cfb" />)

**Quick Results:**
- 🎯 Model Accuracy: **80.6%**
- 🔄 Cross-Validation: **79.4%**
- 📊 Best Feature: **alcohol_to_density** (engineered feature - 16.9% importance)
- 💾 Generated Files: `wine_quality_model.pkl`, `scaler.pkl`, `evaluation_report.json`

---

## 📚 Table of Contents

- [Overview](#overview)
- [What This Lab Demonstrates](#what-this-lab-demonstrates)
- [Dataset](#dataset)
- [Pipeline Architecture](#pipeline-architecture)
- [Prerequisites](#prerequisites)
- [Setup Instructions](#setup-instructions)
- [Running the Pipeline](#running-the-pipeline)
- [Understanding the Results](#understanding-the-results)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

This lab demonstrates a complete **MLOps workflow** using **Apache Airflow** to orchestrate a machine learning pipeline. The pipeline predicts wine quality (good vs. bad) based on physicochemical properties using a Random Forest classifier.

### Key Technologies
- **Apache Airflow**: Workflow orchestration
- **Docker**: Containerization
- **Python**: ML development
- **Scikit-learn**: Machine learning
- **Pandas**: Data processing

---

## 🚀 What This Lab Demonstrates

By completing this lab, you'll understand how to:

1. **Orchestrate ML workflows** with Airflow DAGs
2. **Pass data between tasks** using XCom (with pickle serialization)
3. **Containerize ML pipelines** using Docker
4. **Engineer features** for better model performance
5. **Train and evaluate** classification models
6. **Generate automated reports** for model performance
7. **Structure production-ready** ML code

---

## 📊 Dataset

**Source:** [Red Wine Quality Dataset (Kaggle)](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009)

The dataset contains 1,599 samples of Portuguese "Vinho Verde" red wine with 11 physicochemical features:

| Feature | Description |
|---------|-------------|
| `fixed acidity` | Most acids involved with wine (tartaric acid) |
| `volatile acidity` | Amount of acetic acid (too high = vinegar taste) |
| `citric acid` | Adds freshness and flavor |
| `residual sugar` | Sugar remaining after fermentation |
| `chlorides` | Amount of salt |
| `free sulfur dioxide` | Prevents microbial growth and oxidation |
| `total sulfur dioxide` | Free + bound forms of SO2 |
| `density` | Density of wine |
| `pH` | Acidity level (0-14 scale) |
| `sulphates` | Wine additive (antimicrobial and antioxidant) |
| `alcohol` | Alcohol percentage |
| `quality` | Score between 0-10 (target variable) |

**Our Approach:** We convert this into a **binary classification** problem:
- **Good Wine:** Quality ≥ 6
- **Bad Wine:** Quality < 6

---

## 🏗️ Pipeline Architecture

The Airflow DAG orchestrates 4 sequential tasks:

```
┌─────────────────────┐
│  1. Load Data       │  ← Read CSV and explore dataset
└──────────┬──────────┘
           │ (XCom passes serialized data)
           ▼
┌─────────────────────┐
│  2. Feature Eng.    │  ← Create new features, split data, scale
└──────────┬──────────┘
           │ (XCom passes preprocessed arrays)
           ▼
┌─────────────────────┐
│  3. Train Model     │  ← Train Random Forest with CV
└──────────┬──────────┘
           │ (XCom passes training metrics)
           ▼
┌─────────────────────┐
│  4. Evaluate        │  ← Test model, generate report
└─────────────────────┘
```

### Task Details

#### Task 1: Load Wine Data
- Reads the Red Wine Quality CSV file
- Performs initial data exploration
- Checks for missing values
- Analyzes quality distribution
- Serializes data for next task

#### Task 2: Feature Engineering
- Creates binary target variable (good/bad wine)
- Engineers 3 new features:
  - `alcohol_to_density`: Alcohol relative to density
  - `acid_ratio`: Fixed/volatile acidity ratio
  - `sulfur_ratio`: Free/total sulfur dioxide ratio
- Splits data (80% train, 20% test)
- Applies StandardScaler for normalization
- Prevents data leakage (scaler fit only on train)

#### Task 3: Train Random Forest Model
- Configures Random Forest (100 trees, max_depth=10)
- Performs 5-fold cross-validation
- Trains final model on full training set
- Calculates feature importance scores
- Saves model and scaler to disk

#### Task 4: Evaluate Model
- Loads trained model
- Makes predictions on test set
- Calculates comprehensive metrics:
  - Accuracy, Precision, Recall, F1-Score
  - Confusion Matrix
  - Sensitivity & Specificity
- Compares train vs. test performance
- Saves evaluation report as JSON

---

## 🔧 Prerequisites

### Required Software
- **Docker Desktop** (with at least 4GB RAM, ideally 8GB)
  - [Windows Installation](https://docs.docker.com/desktop/install/windows-install/)
  - [Mac Installation](https://docs.docker.com/desktop/install/mac-install/)
  - [Linux Installation](https://docs.docker.com/desktop/install/linux-install/)

### System Requirements
- **OS:** Windows 10/11, macOS, or Linux
- **RAM:** Minimum 4GB allocated to Docker (8GB recommended)
- **Disk Space:** At least 2GB free

---

## ⚙️ Setup Instructions

### Step 1: Clone or Navigate to the Lab Directory

```bash
cd "/Users/mohan/NEU/FALL 2025/MLOps/MLOPS_LABS/MLOps_Labs/Airflow_labs"
```

### Step 2: Start Docker Desktop

Make sure Docker Desktop is running before proceeding.

### Step 3: Download the Docker Compose File

```bash
# For Mac/Linux
curl -LfO 'https://airflow.apache.org/docs/apache-airflow/2.9.2/docker-compose.yaml'

# For Windows (PowerShell)
curl -o docker-compose.yaml https://airflow.apache.org/docs/apache-airflow/2.9.2/docker-compose.yaml
```

### Step 4: Create Required Directories

```bash
# Mac/Linux
mkdir -p ./logs ./plugins ./config

# Windows (Command Prompt)
mkdir logs plugins config

# Windows (PowerShell)
mkdir logs, plugins, config
```

### Step 5: Configure Airflow User

```bash
# Mac/Linux
echo -e "AIRFLOW_UID=$(id -u)" > .env

# Windows - manually create a file named .env with this content:
AIRFLOW_UID=50000
```

### Step 6: Modify docker-compose.yaml

Open `docker-compose.yaml` and update the following sections:

```yaml
# Don't load example DAGs
AIRFLOW__CORE__LOAD_EXAMPLES: 'false'

# Install required Python packages
_PIP_ADDITIONAL_REQUIREMENTS: ${_PIP_ADDITIONAL_REQUIREMENTS:- pandas scikit-learn}

# Add output directory (add this line under volumes section)
- ${AIRFLOW_PROJ_DIR:-.}/working_data:/opt/airflow/working_data

# Change admin credentials (optional but recommended)
_AIRFLOW_WWW_USER_USERNAME: ${_AIRFLOW_WWW_USER_USERNAME:-admin}
_AIRFLOW_WWW_USER_PASSWORD: ${_AIRFLOW_WWW_USER_PASSWORD:-admin}
```

### Step 7: Initialize Airflow Database

```bash
docker compose up airflow-init
```

Wait for the message: `airflow-init_1 exited with code 0`

### Step 8: Start Airflow

```bash
docker compose up
```

Wait until you see:
```
airflow-webserver-1  | 127.0.0.1 - - [DATE] "GET /health HTTP/1.1" 200 318 "-" "curl/7.88.1"
```

### Step 9: Access Airflow Web UI

1. Open your browser and go to: **http://localhost:8080**
2. Login with credentials:
   - **Username:** `admin` (or what you set in docker-compose.yaml)
   - **Password:** `admin` (or what you set in docker-compose.yaml)

---

## 🎮 Running the Pipeline

### Method 1: Manual Trigger (Recommended for Learning)

1. In the Airflow UI, find the DAG named **`Wine_Quality_ML_Pipeline`**
2. Toggle the switch to **ON** (activates the DAG)
3. Click the **▶️ Trigger DAG** button (play icon on the right)
4. Monitor progress in the **Graph** view

### Method 2: Schedule-Based Execution

To run automatically on a schedule, modify the DAG file:

```python
# In wine_quality_dag.py, change:
schedule_interval=None  

# To one of these:
schedule_interval='@daily'    # Run once per day
schedule_interval='@weekly'   # Run once per week
schedule_interval='0 9 * * *' # Run at 9 AM every day (cron format)
```

### Monitoring Task Execution

1. **Graph View:** Visual representation of task dependencies
2. **Grid View:** Task runs over time
3. **Logs:** Click any task → Logs tab to see detailed output

---

## 📈 Understanding the Results

### ✅ Successful Pipeline Execution

![Pipeline Success](assets/pipeline_success.png)

*Screenshot showing all 4 tasks completed successfully in the Airflow UI*

### Where to Find Results

After the pipeline completes successfully:

1. **Model Files** (in `dags/model/`):
   - `wine_quality_model.pkl` - Trained Random Forest model
   - `scaler.pkl` - Feature scaler for preprocessing
   - `evaluation_report.json` - Detailed metrics

2. **Airflow Logs:**
   - Click on the last task (`evaluate_model_task`) → **Logs** tab
   - View comprehensive performance metrics

### 🎯 Actual Results from This Implementation

**Model Performance (October 6, 2025):**
- ✅ **Test Accuracy:** 80.6%
- ✅ **Cross-Validation Score:** 79.4%
- ✅ **Precision (Good Wine):** 83.4%
- ✅ **Recall (Good Wine):** 79.5%
- ✅ **F1-Score:** 81.4%

**Top 5 Most Important Features:**
1. `alcohol_to_density` (16.9%) - Our engineered feature!
2. `alcohol` (11.5%)
3. `sulphates` (11.3%)
4. `volatile acidity` (8.4%)
5. `acid_ratio` (7.7%) - Our engineered feature!

**Confusion Matrix:**
```
                 Predicted Bad  Predicted Good
Actual Bad            122              27
Actual Good           35              136
```

**Key Insights:**
- Model correctly identifies 80.6% of wines
- Engineered features (`alcohol_to_density`, `acid_ratio`) are among top predictors
- Minimal overfitting: CV score (79.4%) ≈ Test score (80.6%)
- Model performs well on both classes (balanced performance)

### Interpreting the Evaluation Report

Open `dags/model/evaluation_report.json`:

```json
{
  "test_accuracy": 0.85,           // Overall accuracy on test set
  "classification_report": {       // Per-class metrics
    "0": {                         // Bad wine class
      "precision": 0.82,
      "recall": 0.78,
      "f1-score": 0.80
    },
    "1": {                         // Good wine class
      "precision": 0.87,
      "recall": 0.90,
      "f1-score": 0.88
    }
  },
  "confusion_matrix": [...],       // Prediction breakdown
  "sensitivity": 0.90,             // True positive rate
  "specificity": 0.78,             // True negative rate
  "top_features": [...]            // Most important features
}
```

### What Good Results Look Like

- **Accuracy > 80%:** Model is performing well
- **Precision & Recall balanced:** Not biased toward one class
- **CV Score ≈ Test Score:** Model generalizes well (no overfitting)
- **Top features make sense:** Domain-relevant features ranked high

---

## 📁 Project Structure

```
Airflow_labs/
├── assets/                          # Screenshots and images
│   └── pipeline_success.png         # Successful execution screenshot
├── dags/
│   ├── wine_quality_dag.py          # Main DAG definition
│   ├── src/
│   │   ├── __init__.py
│   │   └── wine_pipeline.py         # ML pipeline functions
│   ├── data/
│   │   └── Red Wine Quality.csv     # Dataset (1,599 samples)
│   └── model/                       # Generated after pipeline runs ✅
│       ├── wine_quality_model.pkl   # Trained Random Forest (80.6% accuracy)
│       ├── scaler.pkl               # Feature scaler
│       └── evaluation_report.json   # Performance metrics
├── logs/                            # Airflow task logs (auto-generated)
├── plugins/                         # Airflow plugins (optional)
├── config/                          # Airflow configuration (optional)
├── docker-compose.yaml              # Docker services configuration
├── .env                             # Environment variables
├── .gitignore                       # Git ignore rules
├── README.md                        # This file
├── QUICK_START.md                   # One-page setup guide
├── LAB_SUMMARY.md                   # Detailed lab overview
└── requirements.txt                 # Python dependencies

Note: __pycache__ folders are automatically generated and gitignored
```

---

## 🐛 Troubleshooting

### Issue: Docker containers won't start

**Solution:**
```bash
# Stop all containers
docker compose down

# Remove volumes and restart
docker compose down -v
docker compose up airflow-init
docker compose up
```

### Issue: "Port 8080 already in use"

**Solution:**
```bash
# Find and kill process using port 8080
# Mac/Linux:
lsof -ti:8080 | xargs kill

# Windows:
netstat -ano | findstr :8080
taskkill /PID <PID_NUMBER> /F
```

### Issue: DAG not appearing in UI

**Solution:**
1. Check `dags/` folder is correctly mounted in docker-compose.yaml
2. Verify Python syntax: `python dags/wine_quality_dag.py`
3. Check Airflow logs: `docker compose logs airflow-scheduler`

### Issue: Import errors in tasks

**Solution:**
1. Ensure packages are listed in `_PIP_ADDITIONAL_REQUIREMENTS`
2. Restart containers: `docker compose restart`
3. Check if packages installed: `docker compose exec airflow-worker pip list`

### Issue: Out of memory errors

**Solution:**
1. Increase Docker memory allocation (Docker Desktop → Settings → Resources)
2. Reduce dataset size or model complexity

### Issue: XCom data too large

**Solution:**
- XCom has size limits (~1MB by default)
- Our pipeline uses pickle serialization which is efficient
- If issues persist, consider using Airflow's `XCom backend` with external storage

---

## 🎓 Learning Outcomes

After completing this lab, you should understand:

✅ How to design ML workflows as Airflow DAGs  
✅ Task dependencies and data passing with XCom  
✅ Docker containerization for ML pipelines  
✅ Feature engineering best practices  
✅ Model evaluation and performance metrics  
✅ Production-ready code structure  
✅ MLOps workflow orchestration  

---

## 📚 Additional Resources

- [Apache Airflow Documentation](https://airflow.apache.org/docs/)
- [Airflow Best Practices](https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html)
- [Docker Documentation](https://docs.docker.com/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Wine Quality Dataset Paper](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009)

---

## 🤝 Credits

- **Lab Created By:** Mohan Bhosale
- **Course:** MLOps - Fall 2025
- **Dataset Source:** UCI Machine Learning Repository / Kaggle
- **Inspired By:** Original Airflow Lab 1 (K-Means clustering example)

---

## 📝 Notes for Submission

This lab demonstrates:
- Custom dataset selection (Red Wine Quality from Kaggle)
- Different ML algorithm (Random Forest vs. K-Means)
- Enhanced feature engineering
- Comprehensive evaluation metrics
- Production-quality code with detailed comments
- Complete documentation

**Key Differences from Original Lab:**
- Binary classification instead of clustering
- Feature importance analysis
- Cross-validation for robust evaluation
- More detailed logging and explanations
- Business-relevant insights from results

---

## 🔄 Stopping the Pipeline

When you're done:

```bash
# Stop all containers (keeps data)
docker compose stop

# Stop and remove containers (clean slate)
docker compose down

# Remove everything including volumes (complete cleanup)
docker compose down -v
```

---

**Happy Learning! 🚀**

