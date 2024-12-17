import numpy as np

# Backtracking Line Search
def backtracking_line_search(func, grad, curr_point, direction, rho=0.5, c=1e-4):
    # Perform Backtracking Line Search to find the optimal step size.
    step_size = 1.0  # Initial step size
    while func(curr_point + step_size * direction) > (
        func(curr_point) + c * step_size * np.dot(grad, direction)
    ):
        step_size *= rho
    return step_size
