# Distributed Training with Ray - Lab Submission

## Submission Details
- **Name:** Mohan Bhosale 
- **Subject:** MLOps

## Hardware Specifications
This lab was executed on:
- **Device:** Apple MacBook Air (M1 chip)
- **RAM:** 16 GB
- **CPU Cores:** 8 cores (4 performance + 4 efficiency cores)

*Note: Performance results may vary based on hardware specifications. The M1 chip's unified memory architecture and efficient core design contribute to the observed performance improvements.*

## Lab Overview

This lab demonstrates the power of distributed training using **Ray**, a unified framework for scaling AI and Python applications. We compare sequential (traditional) training versus parallel distributed training to showcase the significant performance improvements achievable through distributed computing.

## What is Distributed Training?

Distributed training is a technique that allows machine learning models to be trained across multiple processors, cores, or machines simultaneously. Instead of training models one after another sequentially, distributed training enables parallel execution of multiple training tasks, dramatically reducing the total training time.

### Key Concepts:

1. **Sequential Training (Traditional Approach):**
   - Models are trained one at a time
   - Each model waits for the previous one to complete
   - Utilizes only a single CPU core at a time
   - Simple to implement but inefficient for multiple models

2. **Distributed Training with Ray:**
   - Models are trained simultaneously across available CPU cores
   - Ray automatically manages task scheduling and resource allocation
   - Utilizes all available computational resources
   - Significantly faster for training multiple models

## What We Did in This Lab

### Dataset
We used the **California Housing dataset** from scikit-learn, which contains **20,640 samples** with **8 features**. This is a much larger dataset that better demonstrates the benefits of distributed training. The dataset predicts median house values in California districts based on features like median income, house age, location, etc.

### Experiment Design

1. **Data Preparation:**
   - Loaded the California Housing dataset
   - Split the data into training and testing sets (80/20 split)
   - Prepared the data for model training

2. **Sequential Training (Baseline):**
   - Trained 30 Random Forest Regressor models sequentially
   - Each model used a different number of estimators (increasing from 50 to 340)
   - Models used deeper trees (max_depth=20) and more splits (min_samples_split=5) for increased computational complexity
   - Measured the total wall-clock time for all models
   - This represents the traditional, non-distributed approach

3. **Distributed Training with Ray:**
   - Initialized a Ray cluster to utilize all available CPU cores
   - Placed training data in Ray's distributed object store for efficient sharing
   - Converted the training function to a remote task using `@ray.remote` decorator
   - Trained the same 30 models in parallel across multiple cores
   - Measured the total wall-clock time for comparison

4. **Additional Analysis Steps:**
   - **Performance Comparison:** Calculated speedup ratio between sequential and parallel approaches
   - **Model Evaluation:** Identified the best performing model based on Mean Squared Error (MSE)
   - **Resource Utilization:** Analyzed how Ray distributes tasks across available cores
   - **Visualization:** Created visualizations comparing training times and model performance

### Key Ray Concepts Demonstrated

1. **Ray Remote Functions:**
   - Functions decorated with `@ray.remote` can be executed in parallel
   - Remote functions return object references immediately (non-blocking)
   - Results are retrieved using `ray.get()` (blocking operation)

2. **Object Store:**
   - Ray's distributed memory system
   - `ray.put()` places objects in the object store
   - Objects can be shared across all workers without copying
   - Reduces memory overhead and improves efficiency

3. **Task Scheduling:**
   - Ray automatically schedules tasks across available resources
   - Handles load balancing and resource management
   - Provides observability through the Ray Dashboard

## How It Works

### Sequential Training Flow:
```
Model 1 → Model 2 → Model 3 → ... → Model 30
(Each waits for previous to complete)
Total Time = Sum of all individual training times
```

### Distributed Training Flow:
```
Ray Scheduler
    ├─→ Worker 1: Model 1, Model 5, Model 9, ...
    ├─→ Worker 2: Model 2, Model 6, Model 10, ...
    ├─→ Worker 3: Model 3, Model 7, Model 11, ...
    └─→ Worker 4: Model 4, Model 8, Model 12, ...
(All models train simultaneously)
Total Time ≈ Time of longest individual model
```

### Performance Improvement

The distributed approach achieves significant speedup because:
- **Parallel Execution:** Multiple models train simultaneously instead of sequentially
- **Resource Utilization:** All CPU cores are utilized instead of just one
- **Efficient Data Sharing:** Ray's object store eliminates unnecessary data copying
- **Smart Scheduling:** Ray automatically balances workload across workers

**Expected Results (on M1 MacBook Air with 16 GB RAM):**
- Sequential training: ~5-15 minutes
- Distributed training: ~1-3 minutes
- **Speedup: 3-8x faster** (utilizing 8 CPU cores)
- The larger dataset (20K+ samples) and more complex models (30 models with 50-340 estimators) ensure that the overhead of distributed computing is outweighed by the parallelization benefits

*Note: Actual results may vary based on system load, available memory, and other running processes.*

## Technical Implementation Details

### Ray Initialization
```python
ray.init()  # Starts Ray cluster, detects available resources
```

### Remote Function Definition
```python
@ray.remote
def train_and_score_model(...):
    # Training logic here
    return n_estimators, score
```

### Parallel Execution
```python
# Create list of remote task references
results_ref = [train_and_score_model.remote(...) for j in range(n_models)]

# Retrieve all results (blocks until all complete)
results = ray.get(results_ref)
```

### Object Store Usage
```python
# Place data in object store (shared across workers)
X_train_ref = ray.put(X_train)
X_test_ref = ray.put(X_test)

# Use references in remote functions
train_and_score_model.remote(X_train_ref, X_test_ref, ...)
```

## Benefits of Distributed Training

1. **Faster Training:** Significantly reduces total training time
2. **Scalability:** Can scale from single machine to large clusters
3. **Resource Efficiency:** Better utilization of available hardware
4. **Flexibility:** Easy to add more workers or resources
5. **Fault Tolerance:** Ray can handle worker failures gracefully

## Use Cases

Distributed training is particularly beneficial for:
- **Hyperparameter Tuning:** Training multiple models with different configurations
- **Ensemble Methods:** Training multiple models for ensemble predictions
- **Cross-Validation:** Parallelizing k-fold cross-validation
- **Large-Scale Training:** Training on datasets that don't fit in single machine memory
- **Model Comparison:** Comparing multiple algorithms simultaneously

## Conclusion

This lab successfully demonstrates that distributed training with Ray provides substantial performance improvements over traditional sequential training. By leveraging parallel execution and efficient resource management on an M1 MacBook Air (8 CPU cores, 16 GB RAM), we achieved a significant speedup in training time, making it an essential technique for machine learning workflows that require training multiple models or working with large datasets.

The M1 chip's unified memory architecture and efficient multi-core design make it particularly well-suited for distributed computing tasks, allowing Ray to effectively utilize all available CPU cores for parallel model training.

## Requirements

See `requirements.txt` for all required packages.

## Running the Lab

1. Create and activate virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Open and run `Ray.ipynb` in Jupyter Notebook or JupyterLab

4. Follow the notebook cells sequentially to:
   - Load and prepare the dataset
   - Run sequential training (baseline)
   - Initialize Ray and run distributed training
   - Compare results and analyze performance improvements

