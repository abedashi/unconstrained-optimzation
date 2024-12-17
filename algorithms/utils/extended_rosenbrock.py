import numpy as np

def extended_rosenbrock(x):
    # Calculate the Extended Rosenbrock function.
    n = len(x)
    f_value = 0.0
    for k in range(1, n + 1):
        if k % 2 == 1:  # mod(k, 2) = 1
            f_k = 10 * (x[k - 1] ** 2 - x[k]) if k < n else 0  # Avoid indexing x[k] out of bounds
        else:  # mod(k, 2) = 0
            f_k = x[k - 1] - 1
        f_value += 0.5 * f_k ** 2
    return f_value

def extended_rosenbrock_gradient(x):
    # Calculate the gradient of the Extended Rosenbrock function.
    n = len(x)
    grad = np.zeros(n)
    for k in range(1, n + 1):
        if k % 2 == 1:  # mod(k, 2) = 1
            if k < n:  # For x[k]
                f_k = 10 * (x[k - 1] ** 2 - x[k])
                grad[k - 1] += 10 * 2 * x[k - 1] * f_k  # Derivative wrt x[k-1]
                grad[k] += -10 * f_k  # Derivative wrt x[k]
        else:  # mod(k, 2) = 0
            f_k = x[k - 1] - 1
            grad[k - 1] += f_k  # Derivative wrt x[k-1]
    return grad

def extended_rosenbrock_hessian(x):
    # Calculate the Hessian of the Extended Rosenbrock function.
    n = len(x)
    hessian = np.zeros((n, n))
    for k in range(1, n + 1):
        if k % 2 == 1:  # mod(k, 2) = 1
            if k < n:  # For x[k]
                f_k = 10 * (x[k - 1] ** 2 - x[k])
                hessian[k - 1, k - 1] += 10 * (2 * f_k + 20 * x[k - 1] ** 2)  # Second derivative wrt x[k-1]
                hessian[k - 1, k] += -20 * x[k - 1]  # Mixed derivative wrt x[k-1] and x[k]
                hessian[k, k - 1] += -20 * x[k - 1]  # Mixed derivative wrt x[k] and x[k-1]
                hessian[k, k] += 10  # Second derivative wrt x[k]
        else:  # mod(k, 2) = 0
            hessian[k - 1, k - 1] += 1  # Second derivative wrt x[k-1]
    return hessian
