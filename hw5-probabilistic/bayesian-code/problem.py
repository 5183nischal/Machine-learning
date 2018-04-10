from __future__ import division

import matplotlib.pyplot as plt
import numpy.matlib as matlib
from scipy.stats import multivariate_normal
import numpy as np
import support_code
import math
from numpy.linalg import inv
from sklearn.linear_model import Ridge

def likelihood_func(w, X, y_train, likelihood_var):
    '''
    Implement likelihood_func. This function returns the data likelihood
    given f(y_train | X; w) ~ Normal(Xw, likelihood_var).

    Args:
        w: Weights
        X: Training design matrix with first col all ones (np.matrix)
        y_train: Training response vector (np.matrix)
        likelihood_var: likelihood variance

    Returns:
        likelihood: Data likelihood (float)
    '''

    #TO DO

    #change to matrix 
    X = np.matrix(X)
    y = np.matrix(y_train)

    n = len(y)

    for i in range(n):
        y_hat = np.transpose(w).dot(np.transpose(X[i,:]))
        temp = y[i]*(y_hat - np.logaddexp(0,y_hat))
        temp += (1-y[i])*(y_hat + (y_hat - np.logaddexp(0,y_hat)))

    likelihood = math.exp(temp)

    return likelihood

def get_posterior_params(X, y_train, prior, likelihood_var = 0.2**2):
    '''
    Implement get_posterior_params. This function returns the posterior
    mean vector \mu_p and posterior covariance matrix \Sigma_p for
    Bayesian regression (normal likelihood and prior).

    Note support_code.make_plots takes this completed function as an argument.

    Args:
        X: Training design matrix with first col all ones (np.matrix)
        y_train: Training response vector (np.matrix)
        prior: Prior parameters; dict with 'mean' (prior mean np.matrix)
               and 'var' (prior covariance np.matrix)
        likelihood_var: likelihood variance- default (0.2**2) per the lecture slides

    Returns:
        post_mean: Posterior mean (np.matrix)
        post_var: Posterior mean (np.matrix)
    '''

    # TO DO

    #sigma^2 is likelihood variance
    prior_mean = prior['mean']
    prior_var = prior['var']

    #TO FIND POST-MEAN
    in_matrix = X.T.dot(X) + likelihood_var*prior_var.getI()
    out_matrix = (in_matrix.getI()).dot(X.T)
    post_mean = out_matrix.dot(y_train)

    #to find post-var
    in_matrix = (1/likelihood_var)*(X.T).dot(X) + prior_var.getI()
    post_var = in_matrix.getI()


    return post_mean, post_var

def get_predictive_params(X_new, post_mean, post_var, likelihood_var = 0.2**2):
    '''
    Implement get_predictive_params. This function returns the predictive
    distribution parameters (mean and variance) given the posterior mean
    and covariance matrix (returned from get_posterior_params) and the
    likelihood variance (default value from lecture).

    Args:
        X_new: New observation (np.matrix object)
        post_mean, post_var: Returned from get_posterior_params
        likelihood_var: likelihood variance (0.2**2) per the lecture slides

    Returns:
        - pred_mean: Mean of predictive distribution
        - pred_var: Variance of predictive distribution
    '''

    # TO DO

    pred_mean = (post_mean.T).dot(X_new)
    #pred-variance
    in_matrix = (X_new.T).dot(post_var)
    pred_var = in_matrix.dot(X_new) + likelihood_var

    return pred_mean, pred_var

if __name__ == '__main__':

    '''
    If your implementations are correct, running
        python problem.py
    inside the Bayesian Regression directory will, for each sigma in sigmas_to-test generates plots
    '''

    np.random.seed(46134)
    actual_weights = np.matrix([[0.3], [0.5]])
    data_size = 40
    noise = {"mean":0, "var":0.2 ** 2}
    likelihood_var = noise["var"]
    xtrain, ytrain = support_code.generate_data(data_size, noise, actual_weights)


    #Question (b)
    sigmas_to_test = [1/2]
    for sigma_squared in sigmas_to_test:
        prior = {"mean":np.matrix([[0], [0]]),
                 "var":matlib.eye(2) * sigma_squared}

        post_mean, post_var = get_posterior_params(xtrain, ytrain, prior)
        print(post_mean)
        #ridge regression
        clf = Ridge(alpha=2*0.04)
        clf.fit(np.matrix(xtrain), np.matrix(ytrain))
        print(clf.coef_)

        '''
        support_code.make_plots(actual_weights,
                                xtrain,
                                ytrain,
                                likelihood_var,
                                prior,
                                likelihood_func,
                                get_posterior_params,
                                get_predictive_params)
        '''
    