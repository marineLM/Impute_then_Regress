"""This file contains the following estimators:
    - OracleImputeMLPPytorch
    - ImputeMLPPytorch
"""
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from mlp_new import MLP_reg


class OracleImputeMLPPytorch(BaseEstimator):
    """Imputes using an oracle and then runs a MLP on the imputed data.
       The oracle is for Gaussian data with M(C)AR or Gaussian self-masking
       missing data mechanisms.

    Parameters
    ----------

    add_mask: bool
        Whether or not to concatenate the mask with the data.

    mdm: str
        The missing data mechanism: either 'MCAR', 'MAR' or 'gaussian_sm'.

    mu: array-like, shape (n_features, )
        Mean of the Gaussian distribution.

    Sigma: array-like, shape (n_features, n_features)
        Covariance matrix of the Gaussian distribution.

    tmu: array-like, shape (n_features, )
        Mean of the Gaussian self-masking distribution. Only used if
        `mdm=gaussian_sm'.

    tsigma2: array-like, shape (n_features, )
        Variances of the Gaussian self-masking distributions. Only used if
        `mdm=gaussian_sm'.

    est_params: dict
        The dictionary containing the parameters for the MLP.
    """

    def __init__(self, add_mask, mdm, mu, Sigma, tmu=None, tsigma2=None,
                 **mlp_params):

        self.add_mask = add_mask
        self.mdm = mdm
        self.mu = mu
        self.Sigma = Sigma
        self.tmu = tmu
        self.tsigma2 = tsigma2
        self.mlp_params = mlp_params

        self._reg = MLP_reg(is_mask=add_mask, **self.mlp_params)

    def oracle_impute(self, X):
        T = X.copy()
        for t in T:
            m = np.isnan(t)
            obs = np.where(~m)[0]
            mis = np.where(m)[0]

            sigma_obs = self.Sigma[np.ix_(obs, obs)]
            sigma_obs_inv = np.linalg.inv(sigma_obs)
            sigma_misobs = self.Sigma[np.ix_(mis, obs)]
            sigma_mis = self.Sigma[np.ix_(mis, mis)]

            mu_mis_obs = self.mu[mis] + sigma_misobs.dot(
                sigma_obs_inv).dot(t[obs] - self.mu[obs])

            if self.mdm in ['MCAR', 'MAR']:
                t[mis] = mu_mis_obs

            elif self.mdm == 'gaussian_sm':
                sigma_mis_obs = sigma_mis - \
                    sigma_misobs.dot(sigma_obs_inv).dot(sigma_misobs.T)
                sigma_mis_obs_inv = np.linalg.inv(sigma_mis_obs)

                D_mis_inv = np.diag(1/self.tsigma2[mis])

                S = np.linalg.inv(D_mis_inv + sigma_mis_obs_inv)
                s = S.dot(D_mis_inv.dot(self.tmu[mis]) +
                          sigma_mis_obs_inv.dot(mu_mis_obs))

                t[mis] = s

            else:
                raise ValueError('`mdm`shouyld be one of `MCAR`, `MAR`, or' +
                                 '`gaussian_sm`')
        return T

    def fit(self, X, y, X_val=None, y_val=None):
        M = np.isnan(X)
        M_val = np.isnan(X_val)
        T = self.oracle_impute(X)
        T_val = self.oracle_impute(X_val)
        if self.add_mask:
            T = np.hstack((T, M))
            T_val = np.hstack((T_val, M_val))
        self._reg.fit(T, y, X_val=T_val, y_val=y_val)
        return self

    def predict(self, X):
        M = np.isnan(X)
        T = self.oracle_impute(X)
        if self.add_mask:
            T = np.hstack((T, M))
        return self._reg.predict(T)


class ImputeMLPPytorch(BaseEstimator):
    """Imputes and then runs a MLP (Pytorch based, same as for NeuMiss)
    on the imputed data.

    Parameters
    ----------

    add_mask: bool
        Whether or not to concatenate the mask with the data.

    imputation_type: str
        One of 'mean' or 'MICE'.

    est_params: dict
        The dictionary containing the parameters for the MLP.
    """

    def __init__(self, add_mask, imputation_type, **mlp_params):

        self.add_mask = add_mask
        self.imputation_type = imputation_type
        self.mlp_params = mlp_params

        if self.imputation_type == 'mean':
            self._imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        elif self.imputation_type == 'MICE':
            self._imp = IterativeImputer(random_state=0)

        self._reg = MLP_reg(is_mask=add_mask, **self.mlp_params)

    def concat_mask(self, X, T):
        M = np.isnan(X)
        T = np.hstack((T, M))
        return T

    def fit(self, X, y, X_val=None, y_val=None):
        T = self._imp.fit_transform(X)
        T_val = self._imp.transform(X_val)
        if self.add_mask:
            T = self.concat_mask(X, T)
            T_val = self.concat_mask(X_val, T_val)
        self._reg.fit(T, y, X_val=T_val, y_val=y_val)
        return self

    def predict(self, X):
        T = self._imp.transform(X)
        if self.add_mask:
            T = self.concat_mask(X, T)
        return self._reg.predict(T)
