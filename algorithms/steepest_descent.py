# algorithms/steepest_descent.py
import numpy as np
from algorithms.utils.line_search import backtracking_line_search

# Steepest Descent method
def steepest_descent(func, grad_func, initial_point, rho, c, tol, max_iter):
    curr_point = initial_point
    iter_count = 0
    path = [curr_point]

    while np.linalg.norm(grad_func(curr_point)) > tol and iter_count < max_iter:
        grad = grad_func(curr_point)
        descent_dir = -grad

        # Determine step size using Backtracking Line Search
        step_size = backtracking_line_search(func, grad, curr_point, descent_dir, rho, c)

        # Update the current point
        curr_point = curr_point + step_size * descent_dir
        path.append(curr_point)
        iter_count += 1

    return curr_point, iter_count, np.array(path)