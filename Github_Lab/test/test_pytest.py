"""
Pytest Test Suite for Calculator Module
Author: Mohan Bhosale
Course: MLOps (IE-7374)

This module contains pytest test cases for the calculator functions.
Tests cover positive, negative, and zero cases for comprehensive validation.
"""

import pytest
from src import calculator


def test_fun1():
    """Test addition function with various inputs."""
    assert calculator.fun1(2, 3) == 5
    assert calculator.fun1(5, 0) == 5
    assert calculator.fun1(-1, 1) == 0
    assert calculator.fun1(-1, -1) == -2


def test_fun2():
    """Test subtraction function with various inputs."""
    assert calculator.fun2(2, 3) == -1
    assert calculator.fun2(5, 0) == 5
    assert calculator.fun2(-1, 1) == -2
    assert calculator.fun2(-1, -1) == 0


def test_fun3():
    """Test multiplication function with various inputs."""
    assert calculator.fun3(2, 3) == 6
    assert calculator.fun3(5, 0) == 0
    assert calculator.fun3(-1, 1) == -1
    assert calculator.fun3(-1, -1) == 1


def test_fun4():
    """Test three-number addition function with various inputs."""
    assert calculator.fun4(2, 3, 5) == 10
    assert calculator.fun4(5, 0, -1) == 4
    assert calculator.fun4(-1, -1, -1) == -3
    assert calculator.fun4(-1, -1, 100) == 98


def test_fun1_error_handling():
    """Test that fun1 raises ValueError for invalid inputs."""
    with pytest.raises(ValueError):
        calculator.fun1("string", 5)
    with pytest.raises(ValueError):
        calculator.fun1(5, "string")


def test_fun2_error_handling():
    """Test that fun2 raises ValueError for invalid inputs."""
    with pytest.raises(ValueError):
        calculator.fun2("string", 5)
    with pytest.raises(ValueError):
        calculator.fun2(5, None)


def test_fun3_error_handling():
    """Test that fun3 raises ValueError for invalid inputs."""
    with pytest.raises(ValueError):
        calculator.fun3([], 5)
    with pytest.raises(ValueError):
        calculator.fun3(5, {})


def test_fun4_error_handling():
    """Test that fun4 raises ValueError for invalid inputs."""
    with pytest.raises(ValueError):
        calculator.fun4("string", 5, 10)
    with pytest.raises(ValueError):
        calculator.fun4(5, "string", 10)
    with pytest.raises(ValueError):
        calculator.fun4(5, 10, "string")


# Parametrized tests for more comprehensive testing
@pytest.mark.parametrize("x, y, expected", [
    (0, 0, 0),
    (10, 5, 15),
    (-10, -5, -15),
    (100, -50, 50),
    (1.5, 2.5, 4.0),
])
def test_fun1_parametrized(x, y, expected):
    """Parametrized test for addition function."""
    assert calculator.fun1(x, y) == expected


@pytest.mark.parametrize("x, y, expected", [
    (10, 5, 5),
    (5, 10, -5),
    (0, 0, 0),
    (-10, -5, -5),
    (100, 50, 50),
])
def test_fun2_parametrized(x, y, expected):
    """Parametrized test for subtraction function."""
    assert calculator.fun2(x, y) == expected


@pytest.mark.parametrize("x, y, expected", [
    (2, 3, 6),
    (0, 100, 0),
    (-2, 3, -6),
    (-2, -3, 6),
    (1.5, 2, 3.0),
])
def test_fun3_parametrized(x, y, expected):
    """Parametrized test for multiplication function."""
    assert calculator.fun3(x, y) == expected


@pytest.mark.parametrize("x, y, z, expected", [
    (1, 2, 3, 6),
    (0, 0, 0, 0),
    (-1, -2, -3, -6),
    (10, -5, 5, 10),
    (1.5, 2.5, 1.0, 5.0),
])
def test_fun4_parametrized(x, y, z, expected):
    """Parametrized test for three-number addition function."""
    assert calculator.fun4(x, y, z) == expected

