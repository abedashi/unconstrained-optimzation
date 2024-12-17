import numpy as np

# Chained Rosenbrock function definition
def chained_rosenbrock(x):
    # Get the length of the input vector x
    n = len(x)
    # Initialize the function value
    f = 0
    for i in range(1, n):
        # Update the function value based on the Chained Rosenbrock formula
        f = f + 100 * ((x[i - 1] ** 2 - x[i]) ** 2) + ((x[i - 1] - 1) ** 2)
    # Return the function value
    return f

# Define the gradient of the Chained Rosenbrock function
def chained_rosenbrock_gradient(x):
    # Get the length of the input vector x
    n = len(x)
    # Initialize the gradient vector
    gradient = np.zeros(n)
    for i in range(1, n):
        # Update the gradient vector based on the Chained Rosenbrock gradient formula
        gradient[i - 1] = 400 * (x[i - 1] ** 2 - x[i]) * x[i - 1] + 2 * (x[i - 1] - 1)
        gradient[i] = -200 * (x[i - 1] ** 2 - x[i])
    # Return the gradient vector
    return gradient

# Define the Hessian of the Chained Rosenbrock function
def chained_rosenbrock_hessian(x):
    n = len(x)
    hessian = np.zeros((n, n))
    for i in range(1, n):
        hessian[i - 1, i - 1] += 1200 * x[i - 1] ** 2 - 400 * x[i] + 2
        hessian[i - 1, i] += -400 * x[i - 1]
        hessian[i, i - 1] += -400 * x[i - 1]
        hessian[i, i] += 200
    return hessian