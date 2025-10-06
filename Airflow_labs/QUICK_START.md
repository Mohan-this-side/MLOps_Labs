# 🚀 Quick Start Guide - Wine Quality Classification Lab

**Author:** Mohan Bhosale

---

## One-Page Setup & Run

### Prerequisites ✅
- Docker Desktop installed and running
- At least 4GB RAM allocated to Docker

---

### Setup Commands (Copy & Paste)

```bash
# 1. Navigate to lab directory
cd "/Users/mohan/NEU/FALL 2025/MLOps/MLOPS_LABS/MLOps_Labs/Airflow_labs"

# 2. Download docker-compose.yaml
curl -LfO 'https://airflow.apache.org/docs/apache-airflow/2.9.2/docker-compose.yaml'

# 3. Create directories
mkdir -p ./logs ./plugins ./config

# 4. Set Airflow user (Mac/Linux)
echo -e "AIRFLOW_UID=$(id -u)" > .env

# Windows users: Create .env file manually with: AIRFLOW_UID=50000

# 5. Initialize Airflow
docker compose up airflow-init

# 6. Start Airflow
docker compose up
```

---

### Configure docker-compose.yaml

Open `docker-compose.yaml` and add/modify:

```yaml
# Find and change these lines:
AIRFLOW__CORE__LOAD_EXAMPLES: 'false'
_PIP_ADDITIONAL_REQUIREMENTS: ${_PIP_ADDITIONAL_REQUIREMENTS:- pandas scikit-learn}
```

---

### Access & Run

1. **Open Browser:** http://localhost:8080
2. **Login:** 
   - Username: `airflow`
   - Password: `airflow`
3. **Find DAG:** `Wine_Quality_ML_Pipeline`
4. **Enable:** Toggle switch to ON
5. **Run:** Click ▶️ Trigger DAG button
6. **Monitor:** Go to Graph view to watch progress

---

### View Results

**During Execution:**
- Click any task → Logs tab to see detailed output

**After Completion:**
- Click `evaluate_model_task` → Logs tab
- See final accuracy and metrics

**Files Generated:**
- `dags/model/wine_quality_model.pkl`
- `dags/model/scaler.pkl`
- `dags/model/evaluation_report.json`

---

### Stop Airflow

```bash
# Stop (keeps data)
docker compose stop

# Stop and remove (clean up)
docker compose down
```

---

### Troubleshooting

| Problem | Solution |
|---------|----------|
| Port 8080 in use | `lsof -ti:8080 \| xargs kill` (Mac/Linux) |
| DAG not showing | Check Python syntax: `python dags/wine_quality_dag.py` |
| Out of memory | Increase Docker RAM in Docker Desktop settings |
| Import errors | Restart: `docker compose restart` |

---

### What The Pipeline Does

```
📊 Load Data (1,599 wine samples)
        ↓
🔧 Engineer Features (3 new features)
        ↓
🌲 Train Random Forest (100 trees, 5-fold CV)
        ↓
📈 Evaluate Model (accuracy, precision, recall)
```

**Expected Runtime:** 2-5 minutes

**Expected Accuracy:** 75-85%

---

### Key Files to Review

1. **`wine_quality_dag.py`** - Airflow DAG definition
2. **`src/wine_pipeline.py`** - ML pipeline functions
3. **`README.md`** - Complete documentation
4. **`LAB_SUMMARY.md`** - What was created and why

---

**That's it! You're ready to run the pipeline! 🎉**

For detailed explanations, see `README.md`

