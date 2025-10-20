# GitHub Actions Lab - Automated Testing with CI/CD

**By Mohan Bhosale**

**Course:** MLOps(IE-7374)  
**Institution:** Northeastern University  
**Semester:** Fall 2025

---

## Table of Contents

1. [Overview](#overview)
2. [What is GitHub Actions](#what-is-github-actions)
3. [Project Structure](#project-structure)
4. [Getting Started](#getting-started)
5. [Calculator Module](#calculator-module)
6. [Testing](#testing)
7. [GitHub Actions Workflows](#github-actions-workflows)
8. [GitHub Platform Setup](#github-platform-setup)
9. [Viewing GitHub Actions Results](#viewing-github-actions-results)
10. [Local Testing](#local-testing)
11. [Troubleshooting](#troubleshooting)
12. [Best Practices](#best-practices)
13. [Additional Resources](#additional-resources)

---

## Overview

This lab demonstrates the implementation of **Continuous Integration/Continuous Deployment (CI/CD)** using **GitHub Actions** for automated testing. The project includes a simple calculator module with comprehensive test suites using both **pytest** and **unittest** frameworks.

### Key Learning Objectives

1. **Virtual Environment Management** - Creating isolated Python environments
2. **GitHub Repository Setup** - Proper project structure and version control
3. **Test-Driven Development** - Writing comprehensive test suites
4. **CI/CD Implementation** - Automating tests with GitHub Actions
5. **Workflow Configuration** - Understanding YAML-based workflows

### Lab Modules

This lab focuses on 5 essential modules:

1. Creating a virtual environment for dependency isolation
2. Creating and configuring a GitHub repository
3. Creating Python source files with error handling
4. Creating test files using pytest and unittest frameworks
5. Implementing GitHub Actions for automated testing

---

## What is GitHub Actions

**GitHub Actions** is a powerful automation and CI/CD platform integrated directly into GitHub repositories. It enables you to automate various workflows and tasks directly within your repository.

### Key Features

- **Automate Workflows**: Run tests, build code, and deploy applications automatically
- **Continuous Integration**: Test every code change before merging
- **Continuous Deployment**: Deploy applications when changes are merged
- **Event-Driven**: Trigger workflows on push, pull requests, issues, and more
- **Cloud-Based**: No need for external CI/CD servers

### How GitHub Actions Work

GitHub Actions work based on three core concepts:

1. **Events**: Activities that trigger workflows (push, pull_request, issues, etc.)
2. **Workflows**: Automated processes defined in YAML files stored in `.github/workflows/`
3. **Jobs**: Sets of steps that execute on the same runner (virtual machine)
4. **Steps**: Individual tasks within a job
5. **Actions**: Reusable units of code (like actions/checkout@v2)

### Workflow Execution Flow

```
Event Occurs (e.g., git push)
    ↓
GitHub Actions Triggered
    ↓
Workflow File Read (.github/workflows/*.yml)
    ↓
Jobs Execute on Runners (Ubuntu, Windows, macOS)
    ↓
Steps Execute Sequentially
    ↓
Results Reported (Success/Failure)
```

---

## Project Structure

```
Github_Lab/
├── .github/
│   └── workflows/
│       ├── pytest_action.yml          # Pytest automation workflow
│       └── unittest_action.yml        # Unittest automation workflow
├── .gitignore                         # Git ignore configuration
├── data/
│   └── __init__.py
├── src/
│   ├── __init__.py
│   └── calculator.py                  # Calculator module with error handling
├── test/
│   ├── __init__.py
│   ├── test_pytest.py                 # Pytest test suite (40 tests)
│   └── test_unittest.py               # Unittest test suite (12 tests)
├── README.md                          # This file
└── requirements.txt                   # Python dependencies
```

---

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git installed on your system
- GitHub account
- Text editor or IDE

### Step 1: Create Virtual Environment

Creating a virtual environment ensures your project dependencies are isolated from the global Python environment. This isolation ensures that your project remains consistent, stable, and free from conflicts with other Python packages.

**On macOS/Linux:**

```bash
# Navigate to the project directory
cd /path/to/Github_Lab

# Create virtual environment
python3 -m venv github_env

# Activate virtual environment
source github_env/bin/activate
```

**On Windows:**

```bash
# Navigate to the project directory
cd \path\to\Github_Lab

# Create virtual environment
python -m venv github_env

# Activate virtual environment
github_env\Scripts\activate
```

After activation, you will see the virtual environment's name in your command prompt or terminal, indicating that you are working within the virtual environment.

### Step 2: Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt

# Verify installation
pip list
```

The `requirements.txt` file contains:
```
pytest>=7.4.0
```

---

## Calculator Module

The `calculator.py` module contains four functions with comprehensive error handling and input validation.

### Function Specifications

#### 1. fun1(x, y) - Addition

**Purpose:** Adds two numbers together.

**Parameters:**
- `x` (int/float): First number
- `y` (int/float): Second number

**Returns:** 
- int/float: Sum of x and y

**Raises:** 
- ValueError: If x or y is not a number

**Example:**
```python
>>> fun1(2, 3)
5
>>> fun1(-1, 1)
0
```

#### 2. fun2(x, y) - Subtraction

**Purpose:** Subtracts y from x.

**Parameters:**
- `x` (int/float): First number
- `y` (int/float): Second number

**Returns:** 
- int/float: Difference of x and y (x - y)

**Raises:** 
- ValueError: If x or y is not a number

**Example:**
```python
>>> fun2(5, 3)
2
>>> fun2(-1, -1)
0
```

#### 3. fun3(x, y) - Multiplication

**Purpose:** Multiplies two numbers together.

**Parameters:**
- `x` (int/float): First number
- `y` (int/float): Second number

**Returns:** 
- int/float: Product of x and y

**Raises:** 
- ValueError: If x or y is not a number

**Example:**
```python
>>> fun3(2, 3)
6
>>> fun3(-1, 1)
-1
```

#### 4. fun4(x, y, z) - Three-Number Addition

**Purpose:** Adds three numbers together.

**Parameters:**
- `x` (int/float): First number
- `y` (int/float): Second number
- `z` (int/float): Third number

**Returns:** 
- int/float: Sum of x, y, and z

**Raises:** 
- ValueError: If any input is not a number

**Example:**
```python
>>> fun4(1, 2, 3)
6
>>> fun4(-1, -1, -1)
-3
```

### Complete Usage Example

```python
from src import calculator

# Basic operations
result1 = calculator.fun1(2, 3)      # Returns 5
result2 = calculator.fun2(5, 3)      # Returns 2
result3 = calculator.fun3(2, 4)      # Returns 8
result4 = calculator.fun4(1, 2, 3)   # Returns 6

# Error handling demonstration
try:
    calculator.fun1("string", 5)
except ValueError as e:
    print(e)  # Output: "Both inputs must be numbers."
```

---

## Testing

This project includes two comprehensive test suites demonstrating different testing frameworks and methodologies.

### Pytest Test Suite

**File:** `test/test_pytest.py`

**Total Tests:** 40

**Features:**
- Simple, intuitive assertion syntax: `assert calculator.fun1(2, 3) == 5`
- Parametrized tests for comprehensive coverage
- Error handling tests using `pytest.raises()`
- Automatic test discovery
- Detailed test reports with XML output

**Test Categories:**

1. **Basic Functionality Tests** (4 tests)
   - test_fun1: Addition with various inputs
   - test_fun2: Subtraction with various inputs
   - test_fun3: Multiplication with various inputs
   - test_fun4: Three-number addition with various inputs

2. **Error Handling Tests** (4 tests)
   - test_fun1_error_handling: Invalid input validation
   - test_fun2_error_handling: Invalid input validation
   - test_fun3_error_handling: Invalid input validation
   - test_fun4_error_handling: Invalid input validation

3. **Parametrized Tests** (32 tests)
   - test_fun1_parametrized: 5 test cases with different inputs
   - test_fun2_parametrized: 5 test cases with different inputs
   - test_fun3_parametrized: 5 test cases with different inputs
   - test_fun4_parametrized: 5 test cases with different inputs

**Running Pytest Tests:**

```bash
# Activate virtual environment first
source github_env/bin/activate  # macOS/Linux
# or
github_env\Scripts\activate     # Windows

# Run all pytest tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest test/test_pytest.py

# Run with coverage report
pytest --cov=src

# Generate XML report (used by GitHub Actions)
pytest --junitxml=pytest-report.xml
```

**Sample Output:**
```
============================= test session starts ==============================
platform darwin -- Python 3.13.3, pytest-8.4.2, pluggy-1.6.0
collected 40 items

test/test_pytest.py::test_fun1 PASSED                                    [  2%]
test/test_pytest.py::test_fun2 PASSED                                    [  5%]
test/test_pytest.py::test_fun3 PASSED                                    [  7%]
test/test_pytest.py::test_fun4 PASSED                                    [ 10%]
test/test_pytest.py::test_fun1_error_handling PASSED                     [ 12%]
...
============================== 40 passed in 0.06s ==============================
```

### Unittest Test Suite

**File:** `test/test_unittest.py`

**Total Tests:** 12

**Features:**
- Class-based test organization using `unittest.TestCase`
- Rich assertion methods: `assertEqual`, `assertRaises`, `assertAlmostEqual`
- setUp/tearDown support for test fixtures
- Standard library (no external dependencies)
- Python's built-in testing framework

**Test Categories:**

1. **Basic Functionality Tests** (4 tests)
   - test_fun1: Addition tests
   - test_fun2: Subtraction tests
   - test_fun3: Multiplication tests
   - test_fun4: Three-number addition tests

2. **Error Handling Tests** (4 tests)
   - test_fun1_error_handling
   - test_fun2_error_handling
   - test_fun3_error_handling
   - test_fun4_error_handling

3. **Floating-Point Tests** (4 tests)
   - test_fun1_with_floats
   - test_fun2_with_floats
   - test_fun3_with_floats
   - test_fun4_with_floats

**Running Unittest Tests:**

```bash
# Activate virtual environment first
source github_env/bin/activate  # macOS/Linux
# or
github_env\Scripts\activate     # Windows

# Run all unittest tests
python -m unittest test.test_unittest

# Run with verbose output
python -m unittest test.test_unittest -v

# Run specific test class
python -m unittest test.test_unittest.TestCalculator

# Run specific test method
python -m unittest test.test_unittest.TestCalculator.test_fun1
```

**Sample Output:**
```
test_fun1 (test.test_unittest.TestCalculator.test_fun1)
Test addition function with various inputs. ... ok
test_fun1_error_handling (test.test_unittest.TestCalculator.test_fun1_error_handling)
Test that fun1 raises ValueError for invalid inputs. ... ok
...
----------------------------------------------------------------------
Ran 12 tests in 0.000s

OK
```

### Pytest vs Unittest Comparison

| Feature | Pytest | Unittest |
|---------|--------|----------|
| **Syntax** | Simple assertions (`assert`) | Method-based (`assertEqual`) |
| **Structure** | Functions or classes | Classes only (inherit from TestCase) |
| **Setup** | Minimal boilerplate | More boilerplate required |
| **Fixtures** | `@pytest.fixture` decorator | `setUp()` and `tearDown()` methods |
| **Parametrization** | Built-in `@pytest.mark.parametrize` | Requires subclassing or loops |
| **Plugins** | Extensive ecosystem | Standard library only |
| **Discovery** | Automatic (test_*.py or *_test.py) | Automatic (test*.py) |
| **Reports** | Rich output formats (HTML, XML, JSON) | Standard text output |
| **Learning Curve** | Easier for beginners | Steeper (OOP concepts required) |
| **Execution** | `pytest` command | `python -m unittest` command |

---

## GitHub Actions Workflows

This project includes two separate workflow files that automatically run tests when code is pushed to the repository.

### Workflow 1: Pytest Automation

**File:** `.github/workflows/pytest_action.yml`

**Workflow Name:** Testing with Pytest

**Triggers:**
- Push to `main` branch
- Push to `master` branch
- Push to branches matching pattern `releases/**`
- Pull requests to `main` or `master` branches
- Excludes `dev` branch

**Job Configuration:**
- **Runs on:** ubuntu-latest (virtual machine)
- **Python Version:** 3.8

**Steps:**

1. **Checkout code**
   - Uses: `actions/checkout@v2`
   - Purpose: Retrieves repository code

2. **Set up Python**
   - Uses: `actions/setup-python@v2`
   - Python Version: 3.8
   - Purpose: Configures Python environment

3. **Install dependencies**
   - Command: `pip install -r requirements.txt`
   - Purpose: Installs pytest and other dependencies

4. **Run tests and generate XML report**
   - Command: `pytest --junitxml=pytest-report.xml`
   - Purpose: Executes all pytest tests and creates XML report
   - Continue on error: false (workflow fails if tests fail)

5. **Upload test results**
   - Uses: `actions/upload-artifact@v2`
   - Artifact name: test-results
   - Purpose: Makes pytest-report.xml available for download

6. **Notify on success/failure**
   - Conditional execution based on test results
   - Provides clear status messages

**Complete Workflow Configuration:**

```yaml
name: Testing with Pytest
run-name: Pytest
on:
  push:
    branches:
      - main
      - master
      - 'releases/**'
    branches-ignore:
      - dev
  pull_request:
    branches:
      - main
      - master
  label:
    types:
      - created
  issues:
    types:
      - opened
      - labeled

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - name: Checkout code
      uses: actions/checkout@v2

    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.8

    - name: Install dependencies
      run: pip install -r requirements.txt

    - name: Run tests and generate XML report
      run: pytest --junitxml=pytest-report.xml
      continue-on-error: false

    - name: Upload test results
      if: always()
      uses: actions/upload-artifact@v2
      with:
        name: test-results
        path: pytest-report.xml

    - name: Notify on success
      if: success()
      run: echo "Tests passed successfully"

    - name: Notify on failure
      if: failure()
      run: echo "Tests failed"
```

### Workflow 2: Unittest Automation

**File:** `.github/workflows/unittest_action.yml`

**Workflow Name:** Python Unittests

**Triggers:**
- Push to `main` branch
- Push to `master` branch
- Pull requests to `main` or `master` branches

**Job Configuration:**
- **Runs on:** ubuntu-latest
- **Python Version:** 3.8

**Steps:**

1. **Checkout code**
   - Uses: `actions/checkout@v2`
   - Purpose: Retrieves repository code

2. **Set up Python**
   - Uses: `actions/setup-python@v2`
   - Python Version: 3.8
   - Purpose: Configures Python environment

3. **Install dependencies**
   - Command: `pip install -r requirements.txt`
   - Purpose: Installs required packages

4. **Run unittests**
   - Command: `python -m unittest test.test_unittest`
   - Purpose: Executes all unittest tests

5. **Notify on success/failure**
   - Conditional execution based on test results
   - Provides clear status messages

**Complete Workflow Configuration:**

```yaml
name: Python Unittests

on:
  push:
    branches:
      - main
      - master
  pull_request:
    branches:
      - main
      - master

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout code
      uses: actions/checkout@v2

    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.8

    - name: Install dependencies
      run: pip install -r requirements.txt

    - name: Run unittests
      run: python -m unittest test.test_unittest

    - name: Notify on success
      if: success()
      run: echo "Unit tests passed successfully"

    - name: Notify on failure
      if: failure()
      run: echo "Unit tests failed"
```

---

## GitHub Platform Setup

### Important: No Manual Configuration Required

**GitHub Actions workflows are automatically detected and executed when the workflow files exist in the `.github/workflows/` directory.** You do NOT need to manually configure anything on the GitHub platform itself.

### How It Works

1. **Workflow Files in Repository**
   - Your workflow YAML files must be in `.github/workflows/` directory
   - GitHub automatically scans this directory when you push code

2. **Automatic Detection**
   - When you push commits to GitHub, it checks for workflow files
   - If workflow triggers match the event (e.g., push to main), the workflow runs

3. **No UI Configuration Needed**
   - Unlike other CI/CD platforms (Jenkins, CircleCI), GitHub Actions requires no setup
   - Everything is defined in YAML files in your repository

### Step-by-Step: Deploying Your Repository

#### Step 1: Create GitHub Repository

1. Open a web browser and go to **GitHub.com**
2. Log in to your GitHub account
3. In the upper right corner, click the **"+"** button
4. Select **"New repository"**
5. Configure repository settings:
   - **Repository name:** `Github_Lab` (or any name you prefer)
   - **Description:** (Optional) "GitHub Actions Lab - Automated Testing with CI/CD"
   - **Visibility:** Choose Public or Private
   - **Initialize repository:** DO NOT check "Initialize with README" (you already have files)
   - Leave other options unchecked
6. Click **"Create repository"** button

#### Step 2: Connect Local Repository to GitHub

After creating the repository, GitHub will show you a page with setup instructions. Follow these commands:

```bash
# Navigate to your project directory
cd /path/to/Github_Lab

# Verify you're in a git repository (should already be initialized)
git status

# Add the GitHub repository as remote origin
git remote add origin https://github.com/YOUR_USERNAME/Github_Lab.git

# Verify remote was added
git remote -v
```

#### Step 3: Push Your Code to GitHub

```bash
# Make sure all changes are committed
git status

# If there are uncommitted changes, commit them
git add .
git commit -m "Complete GitHub Actions Lab implementation"

# Push to main branch
git push -u origin main

# Push to master branch (if you created one)
git push -u origin master
```

#### Step 4: Verify Workflows Triggered

After pushing:

1. Navigate to your repository on GitHub: `https://github.com/YOUR_USERNAME/Github_Lab`
2. Click on the **"Actions"** tab (top navigation bar)
3. You should immediately see:
   - **"Testing with Pytest"** workflow running or completed
   - **"Python Unittests"** workflow running or completed

#### Step 5: What Happens Automatically

When you push code to GitHub:

```
1. GitHub receives your push
   ↓
2. GitHub scans .github/workflows/ directory
   ↓
3. Finds pytest_action.yml and unittest_action.yml
   ↓
4. Checks if push matches workflow triggers (main/master branch)
   ↓
5. Triggers BOTH workflows in parallel
   ↓
6. Each workflow:
   - Spins up Ubuntu virtual machine
   - Checks out your code
   - Installs Python 3.8
   - Installs dependencies
   - Runs tests
   - Reports results
   ↓
7. Shows results in Actions tab
```

### Understanding the GitHub Actions Interface

#### Actions Tab Overview

When you click on the "Actions" tab, you'll see:

**Left Sidebar:**
- List of all workflows in your repository
- "Testing with Pytest"
- "Python Unittests"
- "All workflows" option

**Main Area:**
- List of workflow runs
- Each run shows:
  - Workflow name
  - Commit message that triggered it
  - Branch name
  - Status icon (green checkmark, red X, yellow circle)
  - Timestamp
  - Duration

**Status Indicators:**
- **Green checkmark:** All tests passed successfully
- **Red X:** One or more tests failed
- **Yellow circle:** Workflow is currently running
- **Gray circle:** Workflow was cancelled or skipped

#### Viewing Workflow Details

Click on any workflow run to see:

1. **Summary Page:**
   - Overall status
   - Jobs that ran
   - Artifacts generated
   - Time taken

2. **Jobs Section:**
   - Click on "build" job to see detailed logs

3. **Step-by-Step Logs:**
   - Each step shows command output
   - Expand/collapse individual steps
   - See exact commands executed
   - View test output

4. **Artifacts:**
   - Download pytest-report.xml
   - Available for 90 days (default)

### Viewing Test Results

#### Method 1: Workflow Run Logs

1. Go to Actions tab
2. Click on latest workflow run
3. Click on "build" job
4. Expand "Run tests" step
5. View complete test output

Example of what you'll see:
```
Run pytest --junitxml=pytest-report.xml
============================= test session starts ==============================
platform linux -- Python 3.8.18, pytest-7.4.3, pluggy-1.3.0
collected 40 items

test/test_pytest.py::test_fun1 PASSED                                    [  2%]
test/test_pytest.py::test_fun2 PASSED                                    [  5%]
...
============================== 40 passed in 0.06s ==============================
```

#### Method 2: Download Artifacts

1. Go to workflow run summary page
2. Scroll to "Artifacts" section
3. Click on "test-results" to download
4. Unzip and view pytest-report.xml

### Optional: Branch Protection Rules

To ensure all tests pass before merging pull requests:

1. Go to repository **Settings** tab
2. Click **Branches** in left sidebar
3. Under "Branch protection rules", click **Add rule**
4. Configure:
   - **Branch name pattern:** `main` (or `master`)
   - Check **"Require status checks to pass before merging"**
   - Check **"Require branches to be up to date before merging"**
   - Select status checks:
     - "build" (from pytest workflow)
     - "build" (from unittest workflow)
   - Click **Create** or **Save changes**

Now pull requests cannot be merged unless both workflows pass.

### Optional: Workflow Badges

Add status badges to your README:

1. Go to Actions tab
2. Click on a workflow (e.g., "Testing with Pytest")
3. Click the "..." menu (top right)
4. Select "Create status badge"
5. Copy the Markdown code
6. Paste into your README.md

Example badge:
```markdown
![Testing with Pytest](https://github.com/YOUR_USERNAME/Github_Lab/actions/workflows/pytest_action.yml/badge.svg)
```

---

## Viewing GitHub Actions Results

### Accessing Workflow Runs

1. Navigate to your GitHub repository
2. Click on the **"Actions"** tab in the top navigation
3. You'll see a list of all workflow runs with:
   - Workflow name
   - Commit information
   - Status (Success/Failure/In Progress)
   - Timestamp and duration

### Understanding Status Indicators

- **Green checkmark:** All tests passed successfully
- **Red X:** One or more tests failed
- **Yellow dot:** Workflow is currently running
- **Gray dot:** Workflow was cancelled or skipped

### Viewing Detailed Test Results

**Step 1: Click on Workflow Run**
- Click on any workflow run from the list

**Step 2: View Job Details**
- Click on the "build" job to see detailed logs

**Step 3: Expand Steps**
- Each step in the workflow is collapsible
- Click to expand and view full output
- Look for:
  - "Run tests and generate XML report" (pytest)
  - "Run unittests" (unittest)

**Step 4: Review Test Output**

For pytest, you'll see:
```
Run pytest --junitxml=pytest-report.xml
============================= test session starts ==============================
platform linux -- Python 3.8.18, pytest-7.4.3, pluggy-1.3.0
collected 40 items

test/test_pytest.py ............................                         [ 70%]
test/test_unittest.py ............                                       [100%]

============================== 40 passed in 0.04s ==============================
```

For unittest, you'll see:
```
Run python -m unittest test.test_unittest
............
----------------------------------------------------------------------
Ran 12 tests in 0.000s

OK
```

### Downloading Test Artifacts

1. From the workflow run summary page
2. Scroll down to **Artifacts** section
3. Click on **test-results** to download
4. Unzip the downloaded file
5. Open `pytest-report.xml` to view detailed test results in XML format

### Re-running Workflows

If a workflow fails or you want to re-run it:

1. Click on the failed workflow run
2. Click **Re-run jobs** button (top right)
3. Select **Re-run all jobs** or **Re-run failed jobs**

### Cancelling Workflows

If a workflow is running and you need to cancel it:

1. Click on the running workflow
2. Click **Cancel workflow** button (top right)

---

## Local Testing

Before pushing to GitHub, always test locally to ensure everything works.

### Complete Local Testing Checklist

#### 1. Activate Virtual Environment

```bash
# macOS/Linux
source github_env/bin/activate

# Windows
github_env\Scripts\activate
```

#### 2. Run Pytest Tests

```bash
# Basic test run
pytest

# Verbose output
pytest -v

# With coverage
pytest --cov=src

# Generate XML report (same as GitHub Actions)
pytest --junitxml=pytest-report.xml

# Run only specific tests
pytest test/test_pytest.py::test_fun1
```

#### 3. Run Unittest Tests

```bash
# Basic test run
python -m unittest test.test_unittest

# Verbose output
python -m unittest test.test_unittest -v

# Run specific test class
python -m unittest test.test_unittest.TestCalculator

# Run specific test method
python -m unittest test.test_unittest.TestCalculator.test_fun1
```

#### 4. Verify All Tests Pass

Expected output:
- **Pytest:** `============================== 40 passed in X.XXs ==============================`
- **Unittest:** `Ran 12 tests in X.XXXs` followed by `OK`

#### 5. Check for Errors

If tests fail:
- Read error messages carefully
- Check function implementations
- Verify test expectations
- Debug using print statements or debugger

---

## Troubleshooting

### Common Issues and Solutions

#### 1. Tests Fail Locally But Pass in GitHub Actions (or vice versa)

**Symptoms:**
- Different test results in local vs. GitHub Actions

**Solutions:**
- Ensure same Python version (workflow uses 3.8, check yours with `python --version`)
- Check for environment-specific dependencies
- Verify file paths are relative, not absolute
- Ensure all dependencies are in `requirements.txt`

**Commands to fix:**
```bash
# Check Python version
python --version

# Recreate virtual environment with specific version
python3.8 -m venv github_env
source github_env/bin/activate
pip install -r requirements.txt
```

#### 2. Import Errors in Tests

**Symptoms:**
```
ModuleNotFoundError: No module named 'src'
ImportError: cannot import name 'calculator' from 'src'
```

**Solutions:**
- Verify `__init__.py` files exist in all package directories (src/, test/, data/)
- Check that you're running tests from the project root directory
- Ensure package structure is correct

**Commands to fix:**
```bash
# Verify __init__.py files exist
ls -la src/__init__.py
ls -la test/__init__.py
ls -la data/__init__.py

# Create if missing
touch src/__init__.py
touch test/__init__.py
touch data/__init__.py

# Run tests from project root
cd /path/to/Github_Lab
pytest
```

#### 3. GitHub Actions Workflow Not Triggering

**Symptoms:**
- Pushed code to GitHub but no workflows appear in Actions tab
- Actions tab is empty

**Solutions:**
- Verify workflow files are in `.github/workflows/` directory
- Check YAML syntax using online YAML validator
- Ensure branch names match triggers (main, master)
- Verify the workflows directory path (must be `.github/workflows/`, not `github/workflows/`)

**Commands to verify:**
```bash
# Check workflow files exist
ls -la .github/workflows/

# Should show:
# pytest_action.yml
# unittest_action.yml

# Verify file contents
cat .github/workflows/pytest_action.yml
```

#### 4. Workflow Fails with "pip install" Error

**Symptoms:**
```
ERROR: Could not find a version that satisfies the requirement pytest>=7.4.0
```

**Solutions:**
- Check `requirements.txt` syntax
- Ensure package names are correct
- Verify package versions are available

**Fix requirements.txt:**
```
pytest>=7.4.0
```

#### 5. Permission Denied Errors

**Symptoms:**
```
PermissionError: [Errno 13] Permission denied
```

**Solutions:**
- Check file permissions
- Verify virtual environment activation
- Ensure write access to repository
- On Unix systems, check execute permissions

**Commands to fix:**
```bash
# Fix permissions
chmod +x script_name.py

# Verify virtual environment is activated
which python  # Should show path in github_env/
```

#### 6. Virtual Environment Not Activating

**Symptoms:**
- Commands don't find packages
- `which python` shows system Python

**Solutions:**
- Recreate virtual environment
- Use correct activation command for your OS

**Commands to fix:**
```bash
# Remove old virtual environment
rm -rf github_env

# Create new one
python3 -m venv github_env

# Activate (macOS/Linux)
source github_env/bin/activate

# Activate (Windows)
github_env\Scripts\activate

# Verify
which python  # Should be in github_env/
```

#### 7. YAML Syntax Errors

**Symptoms:**
```
Invalid workflow file: .github/workflows/pytest_action.yml
```

**Solutions:**
- Check indentation (YAML uses spaces, not tabs)
- Verify all keys and values are properly formatted
- Use online YAML validator (yamllint.com)

**Common YAML mistakes:**
```yaml
# WRONG (tabs used)
	steps:
	  - name: Test

# CORRECT (spaces used)
  steps:
    - name: Test

# WRONG (missing colon)
name Testing with Pytest

# CORRECT
name: Testing with Pytest
```

---

## Best Practices

### 1. Version Control

**Commit Frequently:**
- Make small, focused commits
- Write clear, descriptive commit messages
- Commit after each logical change

**Good commit messages:**
```bash
git commit -m "Add error handling to calculator functions"
git commit -m "Implement parametrized tests for edge cases"
git commit -m "Configure GitHub Actions workflows for CI/CD"
```

**Use Branches:**
```bash
# Create feature branch
git checkout -b feature/add-division-function

# Make changes, commit
git add .
git commit -m "Add division function with zero-check"

# Push and create pull request
git push origin feature/add-division-function
```

**Pull Request Workflow:**
- Create branch for new feature
- Make changes and commit
- Push branch to GitHub
- Create pull request
- Wait for automated tests to pass
- Merge after approval

### 2. Testing

**Test-Driven Development (TDD):**
1. Write test first (it will fail)
2. Implement minimal code to pass test
3. Refactor code while keeping tests passing
4. Repeat

**Test Coverage:**
- Aim for high test coverage (80%+)
- Test edge cases (zero, negative, max values)
- Test error conditions
- Test with different data types

**Test Organization:**
```python
# Group related tests
def test_fun1():
    # Test normal cases
    assert calculator.fun1(2, 3) == 5
    # Test edge cases
    assert calculator.fun1(0, 0) == 0
    # Test negative numbers
    assert calculator.fun1(-1, -1) == -2
```

### 3. CI/CD

**Keep Workflows Simple:**
- Each workflow should have a single purpose
- Use clear, descriptive names
- Comment complex configurations

**Workflow Optimization:**
- Use caching for dependencies (speeds up builds)
- Run tests in parallel when possible
- Set appropriate timeout values

**Branch Protection:**
- Require status checks before merging
- Require pull request reviews
- Protect main/master branches

### 4. Documentation

**Code Documentation:**
```python
def fun1(x, y):
    """
    Adds two numbers together.
    
    Args:
        x (int/float): First number.
        y (int/float): Second number.
    
    Returns:
        int/float: Sum of x and y.
    
    Raises:
        ValueError: If x or y is not a number.
    
    Examples:
        >>> fun1(2, 3)
        5
    """
```

**README Documentation:**
- Keep README up to date
- Include setup instructions
- Provide usage examples
- Document dependencies

### 5. Dependencies

**requirements.txt Best Practices:**
- Pin major versions: `pytest>=7.4.0`
- Or pin exact versions for reproducibility: `pytest==7.4.0`
- Keep dependencies minimal
- Document why each dependency is needed

**Virtual Environment:**
- Always use virtual environments
- Never commit virtual environment to git
- Document activation instructions
- Include Python version requirements

---

## Additional Resources

### Official Documentation

- **GitHub Actions Documentation:** https://docs.github.com/en/actions
  - Complete reference for GitHub Actions
  - Workflow syntax
  - Available actions marketplace

- **Pytest Documentation:** https://docs.pytest.org/
  - Getting started guide
  - Parametrization
  - Fixtures
  - Plugins

- **Unittest Documentation:** https://docs.python.org/3/library/unittest.html
  - Python's built-in testing framework
  - TestCase class reference
  - Assertion methods

- **YAML Syntax:** https://yaml.org/
  - YAML specification
  - Syntax reference
  - Examples

### CI/CD Resources

- **CI/CD Best Practices:** https://www.atlassian.com/continuous-delivery/principles/continuous-integration-vs-delivery-vs-deployment
  - Understanding CI/CD concepts
  - Best practices
  - Common patterns

- **GitHub Actions Examples:** https://github.com/actions/starter-workflows
  - Template workflows
  - Common use cases
  - Community workflows

### Tutorial Videos

- **GitHub Actions Tutorial:** https://youtu.be/1Wm6MSqj3ZE
  - Step-by-step walkthrough
  - Practical examples
  - Troubleshooting tips

### Learning Resources

- **Python Testing with pytest:** Book by Brian Okken
- **Effective Python Testing:** Book by Anthon van der Neut
- **GitHub Skills:** https://skills.github.com/
  - Interactive GitHub learning paths
  - Hands-on exercises

---

## Learning Outcomes

After completing this lab, you will be able to:

1. **Create and manage Python virtual environments**
   - Understand dependency isolation
   - Use pip for package management
   - Activate/deactivate environments

2. **Structure Python projects properly**
   - Organize code into modules
   - Create package hierarchies
   - Use `__init__.py` files correctly

3. **Write comprehensive tests**
   - Use pytest framework
   - Use unittest framework
   - Implement parametrized tests
   - Test error conditions

4. **Configure GitHub Actions workflows**
   - Write YAML workflow files
   - Understand triggers and events
   - Configure jobs and steps
   - Use GitHub Actions marketplace

5. **Understand CI/CD automation**
   - Automated testing benefits
   - Continuous integration principles
   - Workflow optimization
   - Branch protection strategies

6. **Use YAML for configuration**
   - YAML syntax and structure
   - Key-value pairs and lists
   - Multi-line strings
   - Nested structures

7. **Interpret test results**
   - Read pytest output
   - Read unittest output
   - Download and analyze artifacts
   - Debug test failures

8. **Apply version control best practices**
   - Meaningful commit messages
   - Branch strategies
   - Pull request workflows
   - Repository organization

---

## Conclusion

This lab successfully demonstrates the power of GitHub Actions for automating testing workflows. By implementing both pytest and unittest frameworks, we have created a robust testing pipeline that ensures code quality and reliability.

### Key Achievements

- **Automated Testing:** Tests run automatically on every code push
- **Multiple Testing Frameworks:** Both pytest and unittest integration
- **Comprehensive Error Handling:** Input validation and error testing
- **Professional Project Structure:** Clean organization and documentation
- **CI/CD Best Practices:** Industry-standard workflow implementation

### Skills Developed

- Virtual environment management
- Test-driven development
- YAML configuration
- Git version control
- Continuous integration/continuous deployment
- Software engineering best practices

The automated workflows provide immediate feedback on code changes, enabling faster development cycles and higher code quality.

---

## Author

**Mohan Bhosale**

**Course:** MLOps (IE-7374)  
**Institution:** Northeastern University  
**Semester:** Fall 2025

---

## License

This project is created for educational purposes as part of the MLOps course at Northeastern University.

---

**End of Documentation**
