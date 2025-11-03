# Dataset Download Instructions

## Telco Customer Churn Dataset

### Source
Download the dataset from Kaggle:
- **URL**: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- **Dataset Name**: Telco Customer Churn

### Steps to Download

1. **Create a Kaggle Account** (if you don't have one):
   - Go to https://www.kaggle.com
   - Sign up for a free account

2. **Download the Dataset**:
   - Navigate to the dataset page: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
   - Click the "Download" button
   - Accept the terms if prompted

3. **Extract the Dataset**:
   - The downloaded file will be a ZIP archive
   - Extract it to get the CSV file

4. **Place the File**:
   - The extracted CSV file should be named: `WA_Fn-UseC_-Telco-Customer-Churn.csv`
   - Place this file in this directory (`Lab2/data/`)
   - After placing, your file structure should be:
     ```
     Lab2/data/WA_Fn-UseC_-Telco-Customer-Churn.csv
     ```

### Alternative: Using Kaggle API

If you prefer using the command line:

```bash
# Install Kaggle API (if not installed)
pip install kaggle

# Configure Kaggle API credentials
# Place your kaggle.json file in ~/.kaggle/ directory

# Download the dataset
kaggle datasets download -d blastchar/telco-customer-churn

# Extract the ZIP file
unzip telco-customer-churn.zip

# Move the CSV file to this directory
mv WA_Fn-UseC_-Telco-Customer-Churn.csv Lab2/data/
```

### File Verification

After placing the file, verify it's correct:
- **Expected file name**: `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- **Expected location**: `Lab2/data/WA_Fn-UseC_-Telco-Customer-Churn.csv`
- **Expected size**: ~700KB (approximately 7,000 rows)

### Dataset Information

- **Rows**: ~7,043 customer records
- **Columns**: 21 features
- **Target Variable**: `Churn` (Yes/No)
- **Task**: Binary classification (predict customer churn)

### Notes

- Make sure the file is in CSV format
- The file uses comma (`,`) as the delimiter
- Some columns may contain spaces in names (e.g., "Phone Service")
- The `TotalCharges` column may contain empty strings that need to be handled

