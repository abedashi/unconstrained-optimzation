import numpy as np

def finite_difference(F, x, h, type='c'):
    # Approximates the Jacobian matrix of F at x using finite difference methods.
    if type not in ['fw', 'c']:
        raise ValueError(f"Invalid finite difference type: {type}. Use 'fw' or 'c'.")

    Fx = np.atleast_1d(F(x))  # Ensure the output of F(x) is an array
    m = len(Fx)
    n = len(x)
    JFx = np.zeros((m, n))

    for i in range(n):
        if type == 'fw':
            xh = x.copy()
            xh[i] += h
            JFx[:, i] = (F(xh) - Fx) / h
        elif type == 'c':
            xh_plus = x.copy()
            xh_minus = x.copy()
            xh_plus[i] += h
            xh_minus[i] -= h
            JFx[:, i] = (F(xh_plus) - F(xh_minus)) / (2 * h)

    return JFx

def finite_difference_gradient(f, x, h, type='c'):
    gradfx = np.zeros_like(x)

    if type not in ['fw', 'c']:
        raise ValueError(f"Invalid finite difference type: {type}. Use 'fw' or 'c'.")

    for i in range(len(x)):
        if type == 'fw':
            xh = x.copy()
            xh[i] += h
            gradfx[i] = (f(xh) - f(x)) / h
        elif type == 'c':
            xh_plus = x.copy()
            xh_minus = x.copy()
            xh_plus[i] += h
            xh_minus[i] -= h
            gradfx[i] = (f(xh_plus) - f(xh_minus)) / (2 * h)

    return gradfx

def finite_difference_hessian(f, x, h):
    n = len(x)
    Hessfx = np.zeros((n, n))

    for j in range(n):
        # Diagonal elements
        xh_plus = x.copy()
        xh_minus = x.copy()
        xh_plus[j] += h
        xh_minus[j] -= h
        Hessfx[j, j] = (f(xh_plus) - 2 * f(x) + f(xh_minus)) / (h ** 2)

        # Off-diagonal elements
        for i in range(j + 1, n):
            xh_plus_ij = x.copy()
            xh_plus_i = x.copy()
            xh_plus_j = x.copy()

            xh_plus_ij[i] += h
            xh_plus_ij[j] += h
            xh_plus_i[i] += h
            xh_plus_j[j] += h

            Hessfx[i, j] = (f(xh_plus_ij) - f(xh_plus_i) - f(xh_plus_j) + f(x)) / (h ** 2)
            Hessfx[j, i] = Hessfx[i, j]  # Symmetry of the Hessian

    return Hessfx


def create_finite_diff_problem(func):
    def gradient(x, h):
        return finite_difference_gradient(func, x, h)

    def hessian(x, h):
        return finite_difference_hessian(func, x, h)

    return func, gradient, hessian