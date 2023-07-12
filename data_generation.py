"""
 Generate source terms and solutions for the differential equation

   u'' = sum(c[k]*phi_k(x))  .


@author: Nicolás Guarín-Zapata
@date: March 2023
"""
import numpy as np
import matplotlib.pyplot as plt


SEED = np.random.seed(69)


def cheb_sample(nsample=50, limits=(0, 1)):
    """Sample an interval using Chebyshev nodes

    Parameters
    ----------
    nsample : int, optional
        Number of points, by default 50.
    limits : tuple, optional
        Interval to sample, by default (0, 1).

    Returns
    -------
    x : ndarray (float)
        Coordinates for the sampled interval.
    """
    theta = np.linspace(0, 2*np.pi, nsample)
    x0, x1 = limits
    x = 0.5*(x0 + x1) + 0.5*(x1 - x0)*np.cos(theta)
    return x


def gen_data(n=50, nsample=200, ndata=2500, sample_type=None):
    """Generate data for a trigonometric polynomial

    Parameters
    ----------
    n : int, optional
        Number of terms to include, by default 50.
    nsample : int, optional
        Number of sampling points in (-1, 1), by default 200.
    ndata : int, optional
        Number of pairs source/solutions generated, by default 2500.
    sample_type : string, optional
        Type of sampling, by default None. The default sampling is
        uniform

    Returns
    -------
    x : ndarray, float
        Coordinates for the evaluation points
    sol : ndarray, float
        Solutions for different sources. Shape ndata by nsample.
    source : ndarray, float
        Sources. Shape ndata by nsample.
    """
    if sample_type == "cheby":
        x = cheb_sample(nsample, (0, 1))
    else:
        x = np.linspace(0, 1, nsample)
    coeff = np.random.normal(0, 1, (ndata, n + 1))
    source = np.zeros((ndata, nsample))
    sol = np.zeros((ndata, nsample))
    for k in range(1, n + 1):
        source += np.outer(coeff[:, k], np.sin(k*np.pi*x))
        sol -= np.outer(coeff[:, k], np.sin(k*np.pi*x))/(k*np.pi)**2

    source = (source.T / np.linalg.norm(coeff, axis=1)).T
    sol = (sol.T / np.linalg.norm(coeff, axis=1)).T
    return x, sol, source


def gen_data_mms(n=50, nsample=200, ndata=2500, sample_type=None,
                 bc=(0.0, 0.0), basis="poly", normalize=False):
    """
    Generate data for using the method of manufactured solutions.

    The data is normalized by the maximum value of the source.

    Parameters
    ----------
    n : int, optional
        Number of terms to include, by default 50.
    nsample : int, optional
        Number of sampling points in (-1, 1), by default 200.
    ndata : int, optional
        Number of pairs source/solutions generated, by default 2500.
    sample_type : string, optional
        Type of sampling, by default None. The default sampling is
        uniform
    bc : tuple (float), optional
        Boundary conditions (left, right), by default both are 0.
    basis : string, optional
        Type of basis used: polynomials or trigonometric.

    Returns
    -------
    x : ndarray, float
        Coordinates for the evaluation points
    sol : ndarray, float
        Solutions for different sources. Shape ndata by nsample.
    source : ndarray, float
        Sources. Shape ndata by nsample.
    """
    if sample_type == "cheby":
        x = cheb_sample(nsample, (-1, 1))
    else:
        x = np.linspace(-1, 1, nsample)
    ua, ub = bc
    if basis == "poly":
        coeff = np.random.normal(0, 1, (ndata, n + 1))
    else:
        n = n//2
        A0 = np.random.normal(0, 1, (ndata))
        Ak = np.random.normal(0, 1, (ndata, n + 1))
        Bk = np.random.normal(0, 1, (ndata, n + 1))
    source = np.zeros((ndata, nsample))
    sol = np.zeros((ndata, nsample))
    if basis == "poly":
        for k in range(1, n + 1):
            sol += np.outer(coeff[:, k], x**k)
            source += np.outer(coeff[:, k], k*(k - 1)*x**k)
    else:
        source += np.outer(A0, np.ones_like(x))
        for k in range(1, n + 1):
            source += np.outer(Ak[:, k], np.cos(k*np.pi*x))
            source += np.outer(Bk[:, k], np.sin(k*np.pi*x))
            sol -= (k*np.pi)**2 * np.outer(Ak[:, k], np.cos(k*np.pi*x))
            sol -= (k*np.pi)**2 * np.outer(Bk[:, k], np.sin(k*np.pi*x))
        
    sol += 0.5*(np.outer(ua - sol[:, 0], 1 - x)
                + np.outer(ub - sol[:, -1], 1 + x))

    # The following normalization would make the boundary conditions
    # to change
    if normalize:
        source = (source.T / np.max(source, axis=1)).T
        sol = (sol.T / np.max(source, axis=1)).T
    return x, sol, source


if __name__== "__main__":

    # Old data generation routine
    x, sol, source = gen_data()
    plt.figure()
    plt.plot(x, sol[0,:] / np.max(sol[0,:]))
    plt.plot(x, source[0,:] / np.max(source[0,:]))

    # Manufactured solutions approach
    x, sol, source = gen_data_mms(basis="trig")
    plt.figure()
    plt.plot(x, sol[0,:] / np.max(sol[0,:]))
    plt.plot(x, source[0,:] / np.max(source[0,:]))

    x, sol, source = gen_data_mms(basis="trig", bc=(-1, 5))
    plt.figure()
    plt.plot(x, sol[0,:])
    plt.ylim(-1.2, 5.2)

    plt.show()
