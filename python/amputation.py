'''This file implements amputation procedures according to various missing
data mechanisms. It was inspired from
https://github.com/BorisMuzellec/MissingDataOT/blob/master/utils.py
'''

import numpy as np
from sklearn.utils import check_random_state
from scipy.optimize import fsolve


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def MCAR(X, p, random_state):
    """
    Missing completely at random mechanism.
    Parameters
    ----------
    X : array-like, shape (n, d)
        Data for which missing values will be simulated.
    p : float
        Proportion of missing values to generate for variables which will have
        missing values.
    random_state: int, RandomState instance or None, optional, default None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    Returns
    -------
    mask : array-like, shape (n, d)
        Mask of generated missing values (True if the value is missing).
    """

    rng = check_random_state(random_state)

    n, d = X.shape
    mask = np.zeros((n, d))

    ber = rng.rand(n, d)
    mask = ber < p

    return mask


def MAR_logistic(X, p, p_obs, random_state):
    """
    Missing at random mechanism with a logistic masking model. First, a subset
    of variables with *no* missing values is randomly selected. The remaining
    variables have missing values according to a logistic model with random
    weights, re-scaled so as to attain the desired proportion of missing values
    on those variables.
    Parameters
    ----------
    X : array-like, shape (n, d)
        Data for which missing values will be simulated.
    p : float
        Proportion of missing values to generate for variables which will have
        missing values.
    p_obs : float
        Proportion of variables with *no* missing values that will be used for
        the logistic masking model.
    random_state: int, RandomState instance or None, optional, default None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    Returns
    -------
    mask : array-like, shape (n, d)
        Mask of generated missing values (True if the value is missing).
    """

    rng = check_random_state(random_state)

    n, d = X.shape
    mask = np.zeros((n, d))

    # number of variables that will have no missing values
    # (at least one variable)
    d_obs = max(int(p_obs * d), 1)
    # number of variables that will have missing values
    d_na = d - d_obs

    # Sample variables that will all be observed, and those with missing values
    idxs_obs = rng.choice(d, d_obs, replace=False)
    idxs_nas = np.array([i for i in range(d) if i not in idxs_obs])

    # Other variables will have NA proportions that depend on those observed
    # variables, through a logistic model. The parameters of this logistic
    # model are random, and adapted to the scale of each variable.
    # var = np.var(X, axis=0)
    # coeffs = rng.randn(d_obs, d_na)/np.sqrt(var[idxs_obs, None])

    mu = X.mean(0)
    cov = (X-mu).T.dot(X-mu)/n
    cov_obs = cov[np.ix_(idxs_obs, idxs_obs)]
    coeffs = rng.randn(d_obs, d_na)
    v = np.array([coeffs[:, j].dot(cov_obs).dot(
        coeffs[:, j]) for j in range(d_na)])
    steepness = rng.uniform(low=0.1, high=0.5, size=d_na)
    coeffs /= steepness*np.sqrt(v)

    # Rescale the sigmoid to have a desired amount of missing values
    # ps = sigmoid(X[:, idxs_obs].dot(coeffs) + intercepts)
    # ps /= (ps.mean(0) / p)

    # Move the intercept to have the desired amount of missing values
    intercepts = np.zeros((d_na))
    for j in range(d_na):
        w = coeffs[:, j]

        def f(b):
            s = sigmoid(X[:, idxs_obs].dot(w) + b) - p
            return s.mean()

        res = fsolve(f, x0=0)
        intercepts[j] = res[0]

    ps = sigmoid(X[:, idxs_obs].dot(coeffs) + intercepts)
    ber = rng.rand(n, d_na)
    mask[:, idxs_nas] = ber < ps

    return mask


def MNAR_logistic(X, p, random_state):
    """
    Missing not at random mechanism with a logistic self-masking model.
    All variables have missing values according to a logistic model on *all*
    variables with random weights, re-scaled so as to attain the desired
    proportion of missing values. Since missingness depends on all variables,
    it will depend on unobserved values as well.
    Parameters
    ----------
    X : array-like, shape (n, d)
        Data for which missing values will be simulated.
    p : float
        Proportion of missing values to generate for variables which will have
        missing values.
    random_state: int, RandomState instance or None, optional, default None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    Returns
    -------
    mask : array-like, shape (n, d)
        Mask of generated missing values (True if the value is missing).
    """

    rng = check_random_state(random_state)

    n, d = X.shape
    mask = np.zeros((n, d))

    # Other variables will have NA proportions that depend on those observed
    # variables, through a logistic model. The parameters of this logistic
    # model are random, and adapted to the scale of each variable.
    # var = np.var(X, axis=0)
    # coeffs = rng.randn(d, d)/np.sqrt(var[:, None])

    mu = X.mean(0)
    cov = (X - mu).T.dot(X-mu)/n
    coeffs = rng.randn(d, d)
    v = np.array([coeffs[:, j].dot(cov).dot(coeffs[:, j]) for j in range(d)])
    steepness = rng.uniform(low=0.1, high=0.5, size=d)
    coeffs /= steepness*np.sqrt(v)

    # Rescale to have a desired amount of missing values
    # ps = sigmoid(X.dot(coeffs) + intercepts)
    # ps /= (ps.mean(0) / p)

    # Move the intercept to have the desired amount of missing values
    intercepts = np.zeros((d))
    for j in range(d):
        w = coeffs[:, j]

        def f(b):
            s = sigmoid(X.dot(w) + b) - p
            return s.mean()

        res = fsolve(f, x0=0)
        intercepts[j] = res[0]

    ps = sigmoid(X.dot(coeffs) + intercepts)
    ber = rng.rand(n, d)
    mask = ber < ps

    return mask


def MNAR_logistic_uniform(X, p, p_params, random_state):
    """
    Missing not at random mechanism with a logistic masking model.
    First, a subset of variables is randomly selected to be used as inputs in a
    logistic masking mechanism. The remaining variables have missing values
    according to a logistic model with random weights, re-scaled so as to
    attain the desired proportion of missing values on those variables, as in
    the MAR mechanism. The difference is that the input variables are then
    masked with a MCAR mechanism, which makes part of the missingness
    effectively dependent on unobserved variables.
    Parameters
    ----------
    X : array-like, shape (n, d)
        Data for which missing values will be simulated.
    p : float
        Proportion of missing values to generate for variables which will
        have missing values.
    p_params : float
        Proportion of variables that will be used for the logistic masking
        model.
    random_state: int, RandomState instance or None, optional, default None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    Returns
    -------
    mask : array-like, shape (n, d)
        Mask of generated missing values (True if the value is missing).
    """

    rng = check_random_state(random_state)

    n, d = X.shape
    mask = np.zeros((n, d))

    # number of variables used as inputs
    d_params = max(int(p_params * d), 1)
    # number of variables masked with the logistic model
    d_na = d - d_params

    # Sample variables that will be parameters for the logistic regression:
    # select at least one variable
    idxs_params = np.random.choice(d, d_params, replace=False)
    idxs_nas = np.array([i for i in range(d) if i not in idxs_params])

    # Other variables will have NA proportions that depend on those observed
    # variables, through a logistic model. The parameters of this logistic
    # model are random, and adapted to the scale of each variable.
    # var = np.var(X, axis=0)
    # coeffs = rng.randn(d_params, d_na)/np.sqrt(var[idxs_params, None])

    mu = X.mean(0)
    cov = (X - mu).T.dot(X-mu)/n
    cov_params = cov[np.ix_(idxs_params, idxs_params)]
    coeffs = rng.randn(d_params, d_na)
    v = np.array([coeffs[:, j].dot(cov_params).dot(
        coeffs[:, j]) for j in range(d_na)])
    steepness = rng.uniform(low=0.1, high=0.5, size=d_na)
    coeffs /= steepness*np.sqrt(v)

    # Rescale to have a desired amount of missing values
    # ps = sigmoid(X[:, idxs_params].dot(coeffs) + intercepts)
    # ps /= (ps.mean(0) / p)

    # Move the intercept to have the desired amount of missing values
    intercepts = np.zeros((d_na))
    for j in range(d_na):
        w = coeffs[:, j]

        def f(b):
            s = sigmoid(X[:, idxs_params].dot(w) + b) - p
            return s.mean()

        res = fsolve(f, x0=0)
        intercepts[j] = res[0]

    ps = sigmoid(X[:, idxs_params].dot(coeffs) + intercepts)
    ber = rng.rand(n, d_na)
    mask[:, idxs_nas] = ber < ps

    # Now, mask some values used in the logistic model at random
    # This makes the missingness of other variables potentially dependent on
    # masked values
    mask[:, idxs_params] = rng.rand(n, d_params) < p

    return mask
