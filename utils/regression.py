import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import sympy as sp

def polynomial_regression(x, y, degree, display=False):
    """
    Perform polynomial regression on given x and y data.
    """

    x = np.asarray(x, dtype=np.float64).reshape(-1, 1)
    y = np.asarray(y, dtype=np.float64)

    # Polynomial transformation
    poly = PolynomialFeatures(degree=degree)
    x_poly = poly.fit_transform(x)

    # Fit the linear regression model
    model = LinearRegression()
    model.fit(x_poly, y)

    # Predict and calculate R-squared
    y_pred = model.predict(x_poly)
    r2 = r2_score(y, y_pred)

    # Build the polynomial equation using sympy
    x_symbol = sp.symbols('x')
    polynom = model.intercept_
    for i in range(1, degree + 1):
        polynom += model.coef_[i] * (x_symbol ** i)

    if False and display:
        print(f"Polynomial Equation: {polynom}")
        print(f"R^2: {r2:.4f}")

        # Plot disabled for headless runs
        # plt.scatter(x, y, color='red', label='Data')
        # x_range = np.linspace(np.min(x), np.max(x), 100).reshape(-1, 1)
        # x_range_poly = poly.transform(x_range)
        # y_range_pred = model.predict(x_range_poly)
        # plt.plot(x_range, y_range_pred, color='blue', label=f'Polynomial Fit (Degree {degree})')
        # plt.xlabel('x')
        # plt.ylabel('y')
        # plt.title(f'Polynomial Regression (Degree {degree})')
        # plt.legend()
        # plt.show()

    return polynom, r2

def multi_regression(x, y, target_r2, display=False):
    """
    Perform multi-step regression starting from a linear log regression, then polynomial regression
    to reach the desired R-squared.

    Parameters:
    - x: array-like, independent variable data.
    - y: array-like, dependent variable data.
    - target_r2: float, target R-squared value to reach.

    Returns:
    - final_polynom: sympy expression of the final polynomial equation.
    - final_r2: float, final R-squared score.
    """
    # Logarithmic transformation for initial linear fit
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    X = np.vstack([np.log(x), np.ones(len(x))]).T

    # Initial linear regression using least squares
    m, b = np.linalg.lstsq(X, y, rcond=None)[0]
    y_log_pred = m * np.log(x) + b
    initial_r2 = r2_score(y, y_log_pred)

    if initial_r2 >= target_r2:
        print(f"Linear Log Regression R^2: {initial_r2:.4f}")
        return sp.symbols('x') * m + b, initial_r2

    # Polynomial regression loop to meet the target R-squared
    degree = 2
    polynom, r2 = polynomial_regression(x, y, degree, display=False)

    while r2 < target_r2:
        degree += 1
        polynom, r2 = polynomial_regression(x, y, degree, display=False)

    # Display the final polynomial and plot once target R^2 is reached
    print(f"Final Polynomial Equation: {polynom}")
    print(f"Final R^2: {r2:.4f}")

    if False and display:
        # Plot final result disabled for headless runs
        poly = PolynomialFeatures(degree=degree)
        x_poly = poly.fit_transform(x.reshape(-1, 1))
        model = LinearRegression()
        model.fit(x_poly, y)

        # plt.scatter(x, y, color='red', label='Data')
        # x_range = np.linspace(np.min(x), np.max(x), 100).reshape(-1, 1)
        # x_range_poly = poly.transform(x_range)
        # y_range_pred = model.predict(x_range_poly)
        # plt.plot(x_range, y_range_pred, color='blue', label=f'Polynomial Fit (Degree {degree})')
        # plt.xlabel('x')
        # plt.ylabel('y')
        # plt.title(f'Polynomial Regression (Degree {degree})')
        # plt.legend()
        # plt.show()

    return polynom, r2

def evaluate_polynomial(x_vals, polynomial):
    """
    Evaluates the given polynomial at specified x-values.

    Parameters:
    - x_vals: A scalar or list of x-values.
    - polynomial: A sympy polynomial expression.

    Returns:
    - List of evaluated polynomial values.
    """
    x = sp.Symbol('x')

    # Ensure x_vals is iterable
    if not hasattr(x_vals, '__iter__'):
        x_vals = [x_vals]

    # Evaluate the polynomial for each x-value
    results = [polynomial.subs(x, val).evalf() for val in x_vals]

    return results

def first_minimum_in_interval(polynomial, a, b):
    """
    Finds the first local minimum of a polynomial within the interval [a, b].
    """
    x = sp.symbols('x')

    # Ensure the polynomial has at least degree 2
    if sp.degree(polynomial, gen=x) < 2:
        return -1

    # Compute first and second derivatives
    first_derivative = sp.diff(polynomial, x)
    second_derivative = sp.diff(first_derivative, x)

    # Find numerical roots (critical points) of the first derivative
    critical_points = sp.nroots(first_derivative, n=15)

    # Filter real roots within [a, b]
    real_roots = sorted([
        float(r.evalf()) for r in critical_points
        if abs(sp.im(r)) < 1e-10 and a <= r.evalf() <= b
    ])

    # Return the first critical point that is a local minimum
    for r in real_roots:
        if second_derivative.subs(x, r).evalf() > 0:
            return r

    return -1

def optimal_point(x, y, min_r2=0.8, target_count=10, start_degree=2, max_degree=20, min_improvement=0.003):
    """
    Estimate the optimal x (e.g., a minimum) by fitting polynomials and selecting those
    with sufficient R² and meaningful improvement over the previous degree.
    """
    from statistics import mean

    minima = []
    prev_r2 = 0
    degree = start_degree

    while degree <= max_degree and len(minima) < target_count:
        poly, r2 = polynomial_regression(x, y, degree, display=False)
        improvement = r2 - prev_r2

        if r2 >= min_r2 and improvement >= min_improvement:
            minimum_x = first_minimum_in_interval(poly, x[0], x[-1])
            if minimum_x != -1:
                minima.append(minimum_x)

        prev_r2 = r2
        degree += 1

    if minima:
        return mean(minima)
    else:
        raise ValueError("No valid minima found with sufficient R² improvement.")
