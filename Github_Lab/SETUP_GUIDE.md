# GitHub Actions Lab - Setup & Deployment Guide

**Author:** Mohan Bhosale  
**Course:** MLOps (IE-7374)  
**Date:** October 2025

---

## ✅ Project Status

### Completed Tasks:
- ✅ Complete folder structure created
- ✅ Calculator module with error handling implemented
- ✅ Comprehensive pytest test suite (40 tests)
- ✅ Comprehensive unittest test suite (12 tests)
- ✅ GitHub Actions workflows for pytest and unittest
- ✅ Professional README documentation
- ✅ requirements.txt and .gitignore configured
- ✅ Both `main` and `master` branches created
- ✅ All tests passing locally (40 pytest + 12 unittest = 52 total tests)

---

## 📊 Test Results Summary

### Pytest Results:
```
============================== 40 passed in 0.06s ==============================
```

**Tests Include:**
- 4 basic function tests
- 4 error handling tests
- 32 parametrized tests covering edge cases

### Unittest Results:
```
Ran 12 tests in 0.000s

OK
```

**Tests Include:**
- 4 basic function tests
- 4 error handling tests
- 4 floating-point precision tests

---

## 🚀 Next Steps for Deployment

### Step 1: Push to GitHub

Since git is already initialized and both `main` and `master` branches exist, you can push to GitHub:

```bash
cd "/Users/mohan/NEU/FALL 2025/MLOps/MLOPS_LABS/MLOps_Labs/Github_Lab"

# Push main branch
git push origin main

# Push master branch
git push origin master
```

### Step 2: Verify GitHub Actions

After pushing, GitHub Actions will automatically trigger:

1. Navigate to your repository on GitHub
2. Click on the **"Actions"** tab
3. You should see two workflows running:
   - **Testing with Pytest**
   - **Python Unittests**

### Step 3: View Workflow Results

Each workflow will show:
- ✅ Green checkmark if all tests pass
- ❌ Red X if any tests fail
- 🟡 Yellow dot while running

Click on any workflow run to see detailed logs.

---

## 📂 Project Structure

```
Github_Lab/
├── .github/
│   └── workflows/
│       ├── pytest_action.yml          ✅ Pytest CI/CD workflow
│       └── unittest_action.yml        ✅ Unittest CI/CD workflow
├── .gitignore                         ✅ Git ignore rules
├── data/
│   └── __init__.py                    ✅ Package initialization
├── src/
│   ├── __init__.py                    ✅ Package initialization
│   └── calculator.py                  ✅ Calculator with 4 functions
├── test/
│   ├── __init__.py                    ✅ Package initialization
│   ├── test_pytest.py                 ✅ 40 pytest tests
│   └── test_unittest.py               ✅ 12 unittest tests
├── github_env/                        🚫 Virtual environment (gitignored)
├── README.md                          ✅ Comprehensive documentation
├── requirements.txt                   ✅ pytest>=7.4.0
└── SETUP_GUIDE.md                     ✅ This file
```

---

## 🧪 Local Testing Commands

### Activate Virtual Environment

```bash
# Navigate to project
cd "/Users/mohan/NEU/FALL 2025/MLOps/MLOPS_LABS/MLOps_Labs/Github_Lab"

# Activate environment
source github_env/bin/activate
```

### Run Pytest Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=src

# Generate XML report (as GitHub Actions does)
pytest --junitxml=pytest-report.xml
```

### Run Unittest Tests

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

---

## 🔧 Manual Calculator Testing

You can also test the calculator functions manually:

```bash
# Start Python interpreter
python

# Import calculator
>>> from src import calculator

# Test addition
>>> calculator.fun1(2, 3)
5

# Test subtraction
>>> calculator.fun2(5, 3)
2

# Test multiplication
>>> calculator.fun3(2, 4)
8

# Test three-number addition
>>> calculator.fun4(1, 2, 3)
6

# Test error handling
>>> calculator.fun1("string", 5)
ValueError: Both inputs must be numbers.

# Exit Python
>>> exit()
```

---

## 🌐 GitHub Repository Setup (If Not Already Done)

If you haven't created a GitHub repository yet:

### Option 1: Create New Repository on GitHub

1. Go to [GitHub.com](https://github.com)
2. Click "+" → "New repository"
3. Name: `Github_Lab` or `MLOps-GitHub-Actions-Lab`
4. Choose Public or Private
5. **DO NOT** initialize with README (we already have one)
6. Click "Create repository"

### Option 2: Link Existing Repository

```bash
cd "/Users/mohan/NEU/FALL 2025/MLOps/MLOPS_LABS/MLOps_Labs/Github_Lab"

# Check current remote
git remote -v

# If no remote exists, add one
git remote add origin <your-repository-url>

# Push both branches
git push -u origin main
git push -u origin master
```

---

## 📊 GitHub Actions Workflow Details

### Workflow 1: pytest_action.yml

**Triggers:**
- Push to `main` or `master` branches
- Push to `releases/**` branches
- Pull requests to `main` or `master`
- Excludes `dev` branch

**What It Does:**
1. Checks out code
2. Sets up Python 3.8
3. Installs dependencies
4. Runs pytest with XML report
5. Uploads test results
6. Notifies success/failure

### Workflow 2: unittest_action.yml

**Triggers:**
- Push to `main` or `master` branches
- Pull requests to `main` or `master`

**What It Does:**
1. Checks out code
2. Sets up Python 3.8
3. Installs dependencies
4. Runs unittest tests
5. Notifies success/failure

---

## 🎯 Expected GitHub Actions Behavior

When you push to GitHub:

```
🔄 Git Push → GitHub Repository

↓

🚀 GitHub Actions Triggered

↓

🔀 Two Workflows Run in Parallel:

├── ✅ Testing with Pytest
│   ├── Checkout code
│   ├── Setup Python 3.8
│   ├── Install pytest
│   ├── Run 40 tests
│   ├── Generate pytest-report.xml
│   ├── Upload artifacts
│   └── ✅ Tests passed successfully

└── ✅ Python Unittests
    ├── Checkout code
    ├── Setup Python 3.8
    ├── Install dependencies
    ├── Run 12 tests
    └── ✅ Unit tests passed successfully
```

---

## 🔍 Troubleshooting

### Issue: Tests fail in GitHub Actions but pass locally

**Solution:**
- Check Python version (workflow uses 3.8, you might have 3.13)
- Verify all dependencies are in requirements.txt
- Check for OS-specific code (GitHub Actions uses Ubuntu)

### Issue: Workflow doesn't trigger

**Solution:**
- Verify workflow files are in `.github/workflows/`
- Check YAML syntax (use online YAML validator)
- Ensure branch names match triggers (main, master)

### Issue: Import errors in tests

**Solution:**
- Verify `__init__.py` exists in all packages
- Check that you're running tests from project root
- Use correct import: `from src import calculator`

---

## 📈 Adding More Tests

To add more calculator functions and tests:

### 1. Add New Function to calculator.py

```python
def fun5(x, y):
    """Divide x by y."""
    if not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
        raise ValueError("Both inputs must be numbers.")
    if y == 0:
        raise ValueError("Cannot divide by zero.")
    return x / y
```

### 2. Add Pytest Test

```python
def test_fun5():
    """Test division function."""
    assert calculator.fun5(10, 2) == 5
    assert calculator.fun5(9, 3) == 3
    
def test_fun5_error_handling():
    """Test division by zero error."""
    with pytest.raises(ValueError):
        calculator.fun5(5, 0)
```

### 3. Add Unittest Test

```python
def test_fun5(self):
    """Test division function."""
    self.assertEqual(calculator.fun5(10, 2), 5)
    self.assertEqual(calculator.fun5(9, 3), 3)
```

### 4. Commit and Push

```bash
git add .
git commit -m "Add division function with tests"
git push origin main
```

The workflows will automatically run with the new tests!

---

## 🎓 Key Concepts Demonstrated

### 1. Virtual Environments
- Isolated Python environment
- Project-specific dependencies
- Reproducible setup

### 2. Test-Driven Development
- Write tests first
- Ensure code correctness
- Catch bugs early

### 3. Continuous Integration
- Automated testing on every commit
- Immediate feedback
- Prevent breaking changes

### 4. Dual Testing Frameworks
- Pytest: Modern, simple syntax
- Unittest: Standard library, class-based

### 5. GitHub Actions
- YAML-based workflow configuration
- Event-driven automation
- Cloud-based CI/CD

---

## 📚 Additional Resources

- **GitHub Actions Docs**: https://docs.github.com/en/actions
- **Pytest Docs**: https://docs.pytest.org/
- **Unittest Docs**: https://docs.python.org/3/library/unittest.html
- **YAML Syntax**: https://yaml.org/

---

## ✨ Success Criteria

Your lab is successful when:

- ✅ All 40 pytest tests pass locally
- ✅ All 12 unittest tests pass locally
- ✅ Code pushed to GitHub (main and master branches)
- ✅ Both GitHub Actions workflows show green checkmarks
- ✅ Test artifacts (pytest-report.xml) available for download
- ✅ Clean, professional project structure
- ✅ Comprehensive documentation

---

## 🎉 Conclusion

You have successfully created a comprehensive GitHub Actions lab with:

- **Calculator module** with error handling
- **52 total tests** (40 pytest + 12 unittest)
- **Automated CI/CD** with GitHub Actions
- **Dual workflow setup** for both testing frameworks
- **Professional documentation** and structure

When you push to GitHub, the automated workflows will run, and you'll see the power of CI/CD in action!

**Next Step:** Push your code to GitHub and watch the magic happen! 🚀

---

**Author:** Mohan Bhosale  
**Course:** MLOps (IE-7374)  
**Northeastern University**  
**Fall 2025**

