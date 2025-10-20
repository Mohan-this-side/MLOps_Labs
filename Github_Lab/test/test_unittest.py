"""
Unittest Test Suite for Calculator Module
Author: Mohan Bhosale
Course: MLOps (IE-7374)

This module contains unittest test cases for the calculator functions.
Tests use the unittest framework with class-based test organization.
"""

import sys
import os
import unittest

# Get the path to the project's root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src import calculator


class TestCalculator(unittest.TestCase):
    """Test class for calculator functions using unittest framework."""

    def test_fun1(self):
        """Test addition function with various inputs."""
        self.assertEqual(calculator.fun1(2, 3), 5)
        self.assertEqual(calculator.fun1(5, 0), 5)
        self.assertEqual(calculator.fun1(-1, 1), 0)
        self.assertEqual(calculator.fun1(-1, -1), -2)

    def test_fun2(self):
        """Test subtraction function with various inputs."""
        self.assertEqual(calculator.fun2(2, 3), -1)
        self.assertEqual(calculator.fun2(5, 0), 5)
        self.assertEqual(calculator.fun2(-1, 1), -2)
        self.assertEqual(calculator.fun2(-1, -1), 0)

    def test_fun3(self):
        """Test multiplication function with various inputs."""
        self.assertEqual(calculator.fun3(2, 3), 6)
        self.assertEqual(calculator.fun3(5, 0), 0)
        self.assertEqual(calculator.fun3(-1, 1), -1)
        self.assertEqual(calculator.fun3(-1, -1), 1)

    def test_fun4(self):
        """Test three-number addition function with various inputs."""
        self.assertEqual(calculator.fun4(2, 3, 5), 10)
        self.assertEqual(calculator.fun4(5, 0, -1), 4)
        self.assertEqual(calculator.fun4(-1, -1, -1), -3)
        self.assertEqual(calculator.fun4(-1, -1, 100), 98)

    def test_fun1_error_handling(self):
        """Test that fun1 raises ValueError for invalid inputs."""
        with self.assertRaises(ValueError):
            calculator.fun1("string", 5)
        with self.assertRaises(ValueError):
            calculator.fun1(5, "string")

    def test_fun2_error_handling(self):
        """Test that fun2 raises ValueError for invalid inputs."""
        with self.assertRaises(ValueError):
            calculator.fun2("string", 5)
        with self.assertRaises(ValueError):
            calculator.fun2(5, None)

    def test_fun3_error_handling(self):
        """Test that fun3 raises ValueError for invalid inputs."""
        with self.assertRaises(ValueError):
            calculator.fun3([], 5)
        with self.assertRaises(ValueError):
            calculator.fun3(5, {})

    def test_fun4_error_handling(self):
        """Test that fun4 raises ValueError for invalid inputs."""
        with self.assertRaises(ValueError):
            calculator.fun4("string", 5, 10)
        with self.assertRaises(ValueError):
            calculator.fun4(5, "string", 10)
        with self.assertRaises(ValueError):
            calculator.fun4(5, 10, "string")

    def test_fun1_with_floats(self):
        """Test addition function with floating-point numbers."""
        self.assertAlmostEqual(calculator.fun1(1.5, 2.5), 4.0)
        self.assertAlmostEqual(calculator.fun1(0.1, 0.2), 0.3)

    def test_fun2_with_floats(self):
        """Test subtraction function with floating-point numbers."""
        self.assertAlmostEqual(calculator.fun2(5.5, 2.5), 3.0)
        self.assertAlmostEqual(calculator.fun2(10.7, 5.2), 5.5)

    def test_fun3_with_floats(self):
        """Test multiplication function with floating-point numbers."""
        self.assertAlmostEqual(calculator.fun3(2.5, 2.0), 5.0)
        self.assertAlmostEqual(calculator.fun3(1.5, 3.0), 4.5)

    def test_fun4_with_floats(self):
        """Test three-number addition with floating-point numbers."""
        self.assertAlmostEqual(calculator.fun4(1.5, 2.5, 1.0), 5.0)
        self.assertAlmostEqual(calculator.fun4(0.1, 0.2, 0.3), 0.6)


if __name__ == '__main__':
    unittest.main()

