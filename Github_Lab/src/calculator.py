"""
Calculator Module - GitHub Actions Lab
Author: Mohan Bhosale
Course: MLOps (IE-7374)

This module contains basic calculator functions with proper error handling
and validation for demonstrating automated testing with GitHub Actions.
"""


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
        >>> fun1(-1, 1)
        0
    """
    if not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
        raise ValueError("Both inputs must be numbers.")
    
    return x + y


def fun2(x, y):
    """
    Subtracts two numbers.
    
    Args:
        x (int/float): First number.
        y (int/float): Second number.
    
    Returns:
        int/float: Difference of x and y (x - y).
    
    Raises:
        ValueError: If x or y is not a number.
    
    Examples:
        >>> fun2(5, 3)
        2
        >>> fun2(-1, -1)
        0
    """
    if not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
        raise ValueError("Both inputs must be numbers.")
    
    return x - y


def fun3(x, y):
    """
    Multiplies two numbers together.
    
    Args:
        x (int/float): First number.
        y (int/float): Second number.
    
    Returns:
        int/float: Product of x and y.
    
    Raises:
        ValueError: If either x or y is not a number.
    
    Examples:
        >>> fun3(2, 3)
        6
        >>> fun3(-1, 1)
        -1
    """
    if not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
        raise ValueError("Both inputs must be numbers.")
    
    return x * y


def fun4(x, y, z):
    """
    Adds three numbers together.
    
    Args:
        x (int/float): First number.
        y (int/float): Second number.
        z (int/float): Third number.
    
    Returns:
        int/float: Sum of x, y and z.
    
    Raises:
        ValueError: If any of the inputs is not a number.
    
    Examples:
        >>> fun4(1, 2, 3)
        6
        >>> fun4(-1, -1, -1)
        -3
    """
    if not (isinstance(x, (int, float)) and isinstance(y, (int, float)) and isinstance(z, (int, float))):
        raise ValueError("All inputs must be numbers.")
    
    total_sum = x + y + z
    return total_sum


# Example usage (commented out)
# if __name__ == "__main__":
#     f1_op = fun1(2, 3)
#     f2_op = fun2(2, 3)
#     f3_op = fun3(2, 3)
#     f4_op = fun4(f1_op, f2_op, f3_op)
#     print(f"fun1(2,3) = {f1_op}")
#     print(f"fun2(2,3) = {f2_op}")
#     print(f"fun3(2,3) = {f3_op}")
#     print(f"fun4({f1_op},{f2_op},{f3_op}) = {f4_op}")

