import time
import numpy as np
from algorithms.modified_newton import modified_newton
from algorithms.steepest_descent import steepest_descent
from algorithms.utils.chained_rosenbrock import chained_rosenbrock, chained_rosenbrock_gradient, chained_rosenbrock_hessian
from algorithms.utils.extended_powell_singular import extended_powell_singular, extended_powell_singular_gradient, extended_powell_singular_hessian
from algorithms.utils.rosenbrock import rosenbrock, rosenbrock_grad, rosenbrock_hessian
from algorithms.utils.extended_rosenbrock import extended_rosenbrock, extended_rosenbrock_gradient, extended_rosenbrock_hessian
from algorithms.utils.finite_difference import create_finite_diff_problem

newton, steepest = ["Modified Newton", "Steepest Descent"]
exact, finite_diff = ["exact", "finite_diff"]

test_problems = {
    "Rosenbrock": {
        exact: [rosenbrock, rosenbrock_grad, rosenbrock_hessian],
        finite_diff: create_finite_diff_problem(rosenbrock),
    },
    "Chained Rosenbrock": {
        exact: [chained_rosenbrock, chained_rosenbrock_gradient, chained_rosenbrock_hessian],
        finite_diff: create_finite_diff_problem(chained_rosenbrock),
    },
    "Extended Powell Singular": {
        exact: [extended_powell_singular, extended_powell_singular_gradient, extended_powell_singular_hessian],
        finite_diff: create_finite_diff_problem(extended_powell_singular),
    },
    "Extended Rosenbrock": {
        exact: [extended_rosenbrock, extended_rosenbrock_gradient, extended_rosenbrock_hessian],
        finite_diff: create_finite_diff_problem(extended_rosenbrock),
    },
}

def run_optimization_method(method_name, test_problem, points, tol, max_iter, derivative_type=exact, h=1e-5, rho=0.5, c=1e-4):
    start_time = time.time() # Record the starting time

    # Select the appropriate derivatives
    if derivative_type == exact:
        func, grad, hess = test_problems[test_problem][derivative_type]
    elif derivative_type == finite_diff:
        func, grad_func, hess_func = test_problems[test_problem][derivative_type]
        grad = lambda x: grad_func(x, h)
        hess = lambda x: hess_func(x, h)

    if method_name == newton:
        # Modified Newton requires Hessian
        final_point, num_iters, trajectory = modified_newton(
            func, grad, hess, points, rho, c, tol, max_iter
        )
    elif method_name == steepest:
        # Steepest Descent does not use Hessian
        final_point, num_iters, trajectory = steepest_descent(
            func, grad, points, rho, c, tol, max_iter
        )
    else:
        raise ValueError(f"Invalid method name. Choose '{newton}' or '{steepest}'.")
    
    end_time = time.time()  # Record the end time
    execution_time = end_time - start_time
    
    # Check success/failure
    success = func(final_point) < tol

    # Experimental rate of convergence
    q = 1  # Assuming linear convergence
    rates = []
    for k in range(len(trajectory) - 1):
        numerator = np.linalg.norm(trajectory[k + 1] - final_point)
        denominator = np.linalg.norm(trajectory[k] - final_point) ** q
        if denominator != 0:
            rates.append(numerator / denominator)

    avg_rate_of_convergence = np.mean(rates) if rates else None

    result = {
        "Start Point": points.tolist(),
        "Final Point": final_point.tolist(),
        f"Cost of {test_problem}": func(final_point),
        "Iterations": num_iters,
        "Execution Time (s)": execution_time,
        "Success": success,
        "Gradiant norm": np.linalg.norm(grad(final_point)),
        "Experimental Rate of Convergence": avg_rate_of_convergence,
    }

    return result, trajectory