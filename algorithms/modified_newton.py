# algorithms/modified_newton.py
import numpy as np
from algorithms.utils.line_search import backtracking_line_search

# Modified Newton Method
def modified_newton(func, grad_func, hessian_func, initial_point, rho, c, tol, max_iter, reg_param=1e-4):
    curr_point = initial_point
    iter_count = 0
    path = [curr_point]

    while np.linalg.norm(grad_func(curr_point)) > tol and iter_count < max_iter:
        grad = grad_func(curr_point)
        hessian = hessian_func(curr_point)

        # Regularize the Hessian to ensure positive definiteness
        reg_hessian = hessian + reg_param * np.eye(len(curr_point))

        # Compute the direction of descent
        try:
            descent_dir = np.linalg.solve(reg_hessian, -grad)
        except np.linalg.LinAlgError:
            # Fallback to modified newton descent if Hessian is not invertible
            descent_dir = -grad

        # Determine step size using Backtracking Line Search
        step_size = backtracking_line_search(func, grad, curr_point, descent_dir, rho, c)

        # Update the current point
        curr_point = curr_point + step_size * descent_dir
        path.append(curr_point)
        iter_count += 1

    return curr_point, iter_count, np.array(path)