import numpy as np

# Define the Rosenbrock function
def rosenbrock(point):
    term1 = 100 * (point[1] - point[0] ** 2) ** 2
    term2 = (1 - point[0]) ** 2
    return term1 + term2

# Define the gradient of the Rosenbrock function
def rosenbrock_grad(point):
    grad_x1 = -400 * point[0] * (point[1] - point[0] ** 2) - 2 * (1 - point[0])
    grad_x2 = 200 * (point[1] - point[0] ** 2)
    return np.array([grad_x1, grad_x2])

# Define the Hessian of the Rosenbrock function
def rosenbrock_hessian(point):
    hess_x1_x1 = 1200 * point[0]**2 - 400 * point[1] + 2
    hess_x1_x2 = -400 * point[0]
    hess_x2_x1 = -400 * point[0]
    hess_x2_x2 = 200
    return np.array([[hess_x1_x1, hess_x1_x2], [hess_x2_x1, hess_x2_x2]])