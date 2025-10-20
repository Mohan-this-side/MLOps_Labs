# GitHub Actions Lab - Automated Testing with CI/CD

**By Mohan Bhosale**

---

## 📋 Overview

This lab demonstrates the implementation of **Continuous Integration/Continuous Deployment (CI/CD)** using **GitHub Actions** for automated testing. The project includes a simple calculator module with comprehensive test suites using both **pytest** and **unittest** frameworks.

### Key Learning Objectives:
1. **Virtual Environment Management** - Creating isolated Python environments
2. **GitHub Repository Setup** - Proper project structure and version control
3. **Test-Driven Development** - Writing comprehensive test suites
4. **CI/CD Implementation** - Automating tests with GitHub Actions
5. **Workflow Configuration** - Understanding YAML-based workflows

---

## 🎯 What is GitHub Actions?

**GitHub Actions** is a powerful automation and CI/CD platform integrated directly into GitHub repositories. It enables you to:

- **Automate Workflows**: Run tests, build code, and deploy applications automatically
- **Continuous Integration**: Test every code change before merging
- **Continuous Deployment**: Deploy applications when changes are merged
- **Event-Driven**: Trigger workflows on push, pull requests, issues, and more

### How GitHub Actions Work:

1. **Events**: Activities that trigger workflows (push, pull_request, etc.)
2. **Workflows**: Automated processes defined in YAML files
3. **Jobs**: Sets of steps that execute on the same runner
4. **Steps**: Individual tasks within a job
5. **Actions**: Reusable units of code

---

## 📂 Project Structure

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
│   ├── test_pytest.py                 # Pytest test suite
│   └── test_unittest.py               # Unittest test suite
├── README.md                          # This file
└── requirements.txt                   # Python dependencies
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- GitHub account

### Step 1: Create Virtual Environment

Creating a virtual environment ensures your project dependencies are isolated from the global Python environment.

```bash
# Navigate to the project directory
cd /Users/mohan/NEU/FALL\ 2025/MLOps/MLOPS_LABS/MLOps_Labs/Github_Lab

# Create virtual environment
python3 -m venv github_env

# Activate virtual environment
source github_env/bin/activate      # On macOS/Linux
# or
github_env\Scripts\activate         # On Windows
```

### Step 2: Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt

# Verify installation
pip list
```

---

## 🧮 Calculator Module

The `calculator.py` module contains four functions with comprehensive error handling:

### Functions:

1. **fun1(x, y)** - Addition
   - Returns: x + y
   - Validates input types
   - Raises ValueError for invalid inputs

2. **fun2(x, y)** - Subtraction
   - Returns: x - y
   - Validates input types
   - Raises ValueError for invalid inputs

3. **fun3(x, y)** - Multiplication
   - Returns: x * y
   - Validates input types
   - Raises ValueError for invalid inputs

4. **fun4(x, y, z)** - Three-number addition
   - Returns: x + y + z
   - Validates input types
   - Raises ValueError for invalid inputs

### Example Usage:

```python
from src import calculator

# Basic operations
result1 = calculator.fun1(2, 3)      # Returns 5
result2 = calculator.fun2(5, 3)      # Returns 2
result3 = calculator.fun3(2, 4)      # Returns 8
result4 = calculator.fun4(1, 2, 3)   # Returns 6

# Error handling
try:
    calculator.fun1("string", 5)
except ValueError as e:
    print(e)  # "Both inputs must be numbers."
```

---

## 🧪 Testing

This project includes two comprehensive test suites demonstrating different testing frameworks.

### Pytest Test Suite

**File**: `test/test_pytest.py`

**Features**:
- Simple assertion syntax: `assert calculator.fun1(2, 3) == 5`
- Parametrized tests for comprehensive coverage
- Error handling tests using `pytest.raises()`
- Automatic test discovery
- Detailed test reports

**Running Pytest Tests**:

```bash
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

**Sample Output**:
```
test/test_pytest.py::test_fun1 PASSED                  [ 10%]
test/test_pytest.py::test_fun2 PASSED                  [ 20%]
test/test_pytest.py::test_fun3 PASSED                  [ 30%]
test/test_pytest.py::test_fun4 PASSED                  [ 40%]
test/test_pytest.py::test_fun1_error_handling PASSED   [ 50%]
...
======================== 12 passed in 0.05s =========================
```

### Unittest Test Suite

**File**: `test/test_unittest.py`

**Features**:
- Class-based test organization
- Rich assertion methods: `assertEqual`, `assertRaises`, `assertAlmostEqual`
- setUp/tearDown support for test fixtures
- Standard library (no external dependencies)

**Running Unittest Tests**:

```bash
# Run all unittest tests
python -m unittest test.test_unittest

# Run with verbose output
python -m unittest test.test_unittest -v

# Run specific test class
python -m unittest test.test_unittest.TestCalculator

# Run specific test method
python -m unittest test.test_unittest.TestCalculator.test_fun1
```

**Sample Output**:
```
test_fun1 (test.test_unittest.TestCalculator) ... ok
test_fun2 (test.test_unittest.TestCalculator) ... ok
test_fun3 (test.test_unittest.TestCalculator) ... ok
test_fun4 (test.test_unittest.TestCalculator) ... ok
...
----------------------------------------------------------------------
Ran 12 tests in 0.003s

OK
```

---

## ⚙️ GitHub Actions Workflows

### Workflow 1: Pytest Automation

**File**: `.github/workflows/pytest_action.yml`

**Triggers**:
- Push to `main` or `master` branches
- Push to branches matching `releases/**`
- Pull requests to `main` or `master`
- Excludes `dev` branch

**Steps**:
1. Checkout code from repository
2. Set up Python 3.8 environment
3. Install dependencies from requirements.txt
4. Run pytest tests with XML report generation
5. Upload test results as artifacts
6. Notify on success or failure

**Workflow Configuration**:
```yaml
name: Testing with Pytest
on:
  push:
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
    - name: Run tests
      run: pytest --junitxml=pytest-report.xml
```

### Workflow 2: Unittest Automation

**File**: `.github/workflows/unittest_action.yml`

**Triggers**:
- Push to `main` or `master` branches
- Pull requests to `main` or `master`

**Steps**:
1. Checkout code from repository
2. Set up Python 3.8 environment
3. Install dependencies from requirements.txt
4. Run unittest tests
5. Notify on success or failure

**Workflow Configuration**:
```yaml
name: Python Unittests
on:
  push:
    branches:
      - main
      - master
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - name: Run unittests
      run: python -m unittest test.test_unittest
```

---

## 📊 Pytest vs Unittest Comparison

| Feature | Pytest | Unittest |
|---------|--------|----------|
| **Syntax** | Simple assertions (`assert`) | Method-based (`assertEqual`) |
| **Structure** | Functions or classes | Classes only |
| **Setup** | Minimal | More boilerplate |
| **Fixtures** | `@pytest.fixture` | `setUp()/tearDown()` |
| **Parametrization** | Built-in `@pytest.mark.parametrize` | Requires subclassing |
| **Plugins** | Extensive ecosystem | Standard library |
| **Discovery** | Automatic | Automatic |
| **Reports** | Rich output formats | Standard output |
| **Learning Curve** | Easier | Steeper |

---

## 🔄 GitHub Repository Setup

### Step 1: Create GitHub Repository

1. Go to [GitHub.com](https://github.com)
2. Click "+" → "New repository"
3. Repository name: `Github_Lab` or `MLOps-GitHub-Actions-Lab`
4. Choose visibility (Public/Private)
5. ✅ Initialize with README
6. Click "Create repository"

### Step 2: Initialize Local Repository

```bash
# Navigate to project directory
cd /Users/mohan/NEU/FALL\ 2025/MLOps/MLOPS_LABS/MLOps_Labs/Github_Lab

# Initialize git (if not already initialized)
git init

# Add remote origin
git remote add origin <your-repository-url>

# Create main branch
git checkout -b main

# Add all files
git add .

# Commit changes
git commit -m "Initial commit: GitHub Actions Lab with Calculator and Automated Tests"

# Push to GitHub
git push -u origin main
```

### Step 3: Create Master Branch (if needed)

```bash
# Create master branch from main
git checkout -b master

# Push master branch
git push -u origin master

# Switch back to main
git checkout main
```

---

## ✅ Viewing GitHub Actions Results

### Accessing Workflow Runs:

1. Navigate to your GitHub repository
2. Click on the **"Actions"** tab
3. View all workflow runs with status indicators:
   - ✅ Green checkmark = All tests passed
   - ❌ Red X = Tests failed
   - 🟡 Yellow dot = Workflow running

### Viewing Test Details:

1. Click on a specific workflow run
2. Click on the job name (e.g., "build")
3. Expand steps to see detailed logs
4. Download artifacts (pytest-report.xml) from the artifacts section

### Sample Workflow Run Output:

```
Run pytest --junitxml=pytest-report.xml
============================= test session starts ==============================
platform linux -- Python 3.8.18, pytest-7.4.3, pluggy-1.3.0
collected 12 items

test/test_pytest.py ............                                        [100%]

======================== 12 passed in 0.15s ================================
```

---

## 🎯 Key Benefits of This Implementation

### 1. Automated Testing
- Tests run automatically on every push
- No manual intervention required
- Immediate feedback on code quality

### 2. Early Bug Detection
- Catch errors before they reach production
- Validate changes before merging
- Prevent breaking changes

### 3. Code Quality Assurance
- Enforce testing standards
- Ensure all functions work correctly
- Validate edge cases and error handling

### 4. Team Collaboration
- All team members' code is tested
- Consistent testing across all contributions
- Protected main branch with required checks

### 5. Documentation
- Workflow files serve as documentation
- Clear test coverage visibility
- Easy onboarding for new team members

---

## 🔍 Understanding YAML Syntax

GitHub Actions workflows use YAML (YAML Ain't Markup Language) format:

```yaml
# Comments start with #
key: value                    # Key-value pairs
list:                         # Lists
  - item1
  - item2
nested:                       # Nested structures
  key1: value1
  key2: value2
multiline: |                  # Multiline strings
  This is line 1
  This is line 2
```

---

## 📝 Best Practices

### 1. Version Control
- Commit frequently with meaningful messages
- Use branches for new features
- Create pull requests for code review

### 2. Testing
- Write tests before implementing features (TDD)
- Test edge cases and error conditions
- Maintain high test coverage

### 3. CI/CD
- Keep workflows simple and focused
- Use caching to speed up workflows
- Set up branch protection rules

### 4. Documentation
- Keep README updated
- Document all functions with docstrings
- Include usage examples

---

## 🐛 Troubleshooting

### Common Issues:

**1. Tests fail locally but pass in GitHub Actions (or vice versa)**
- Ensure same Python version locally and in workflow
- Check for environment-specific dependencies
- Verify file paths are relative, not absolute

**2. Import errors in tests**
- Verify `__init__.py` files exist in all packages
- Check PYTHONPATH configuration
- Use relative imports correctly

**3. GitHub Actions workflow not triggering**
- Verify workflow file is in `.github/workflows/`
- Check YAML syntax (use YAML validator)
- Ensure branch names match triggers

**4. Permission denied errors**
- Check file permissions
- Verify virtual environment activation
- Ensure write access to repository

---

## 📚 Additional Resources

- **GitHub Actions Documentation**: https://docs.github.com/en/actions
- **Pytest Documentation**: https://docs.pytest.org/
- **Unittest Documentation**: https://docs.python.org/3/library/unittest.html
- **YAML Syntax**: https://yaml.org/
- **CI/CD Best Practices**: https://www.atlassian.com/continuous-delivery/principles/continuous-integration-vs-delivery-vs-deployment

### Tutorial Videos:
- GitHub Actions Tutorial: https://youtu.be/1Wm6MSqj3ZE

---

## 🎓 Learning Outcomes

After completing this lab, you will understand:

✅ How to create and manage Python virtual environments  
✅ How to structure a Python project with proper testing  
✅ How to write comprehensive tests using pytest and unittest  
✅ How to configure GitHub Actions workflows  
✅ How CI/CD automation improves software development  
✅ How to use YAML for workflow configuration  
✅ How to interpret test results and workflow logs  
✅ Best practices for version control and testing  

---

## 🎉 Conclusion

This lab successfully demonstrates the power of **GitHub Actions** for automating testing workflows. By implementing both **pytest** and **unittest** frameworks, we've created a robust testing pipeline that ensures code quality and reliability.

**Key Achievements**:
- ✅ Automated testing on every code push
- ✅ Multiple testing frameworks integration
- ✅ Comprehensive error handling and validation
- ✅ Professional project structure
- ✅ CI/CD best practices implementation

The automated workflows provide immediate feedback on code changes, enabling faster development cycles and higher code quality! 🚀

---

## 👨‍💻 Author

**Mohan Bhosale**  
Course: MLOps (IE-7374)  
Northeastern University  
Fall 2025

---

## 📄 License

This project is created for educational purposes as part of the MLOps course at Northeastern University.

---

**Happy Testing! 🧪✨**

