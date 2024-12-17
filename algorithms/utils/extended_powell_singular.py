import numpy as np

# Test problem: Extended Powell Singular
def extended_powell_singular(x):
    n = len(x)
    # assert n % 4 == 0, "Dimension must be a multiple of 4."
    return sum(
        (x[4 * i] + 10 * x[4 * i + 1]) ** 2 + 5 * (x[4 * i + 2] - x[4 * i + 3]) ** 2 +
        (x[4 * i + 1] - 2 * x[4 * i + 2]) ** 4 + 10 * (x[4 * i] - x[4 * i + 3]) ** 4 
        for i in range(n // 4)
    )

def extended_powell_singular_gradient(x):
    n = len(x)
    grad = np.zeros_like(x)
    for i in range(n // 4):
        grad[4 * i] = 2 * (x[4 * i] + 10 * x[4 * i + 1]) + 40 * (x[4 * i] - x[4 * i + 3]) ** 3
        grad[4 * i + 1] = 20 * (x[4 * i] + 10 * x[4 * i + 1]) + 4 * (x[4 * i + 1] - 2 * x[4 * i + 2]) ** 3
        grad[4 * i + 2] = 10 * (x[4 * i + 2] - x[4 * i + 3]) - 8 * (x[4 * i + 1] - 2 * x[4 * i + 2]) ** 3
        grad[4 * i + 3] = -10 * (x[4 * i + 2] - x[4 * i + 3]) - 40 * (x[4 * i] - x[4 * i + 3]) ** 3
    return grad

def extended_powell_singular_hessian(x):
    n = len(x)
    hessian = np.zeros((n, n))
    for i in range(n // 4):
        # Extract indices for this group of four variables
        idx1, idx2, idx3, idx4 = 4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3

        # Hessian components for x[idx1]
        hessian[idx1, idx1] = 2 + 120 * (x[idx1] - x[idx4]) ** 2
        hessian[idx1, idx2] = 20
        hessian[idx1, idx4] = -120 * (x[idx1] - x[idx4]) ** 2

        # Hessian components for x[idx2]
        hessian[idx2, idx1] = 20
        hessian[idx2, idx2] = 200 + 12 * (x[idx2] - 2 * x[idx3]) ** 2
        hessian[idx2, idx3] = -24 * (x[idx2] - 2 * x[idx3]) ** 2

        # Hessian components for x[idx3]
        hessian[idx3, idx2] = -24 * (x[idx2] - 2 * x[idx3]) ** 2
        hessian[idx3, idx3] = 10 + 48 * (x[idx2] - 2 * x[idx3]) ** 2
        hessian[idx3, idx4] = -10

        # Hessian components for x[idx4]
        hessian[idx4, idx1] = -120 * (x[idx1] - x[idx4]) ** 2
        hessian[idx4, idx3] = -10
        hessian[idx4, idx4] = 10 + 120 * (x[idx1] - x[idx4]) ** 2

    return hessian