import numpy as np
from sklearn import preprocessing
from scipy.optimize import minimize
from numpy import genfromtxt
import functools
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic


def f_objective(theta, X, y, l2_param):
    '''
    Args:
        theta: 1D numpy array of size num_features
        X: 2D numpy array of size (num_instances, num_features)
        y: 1D numpy array of size num_instances
        l2_param: regularization parameter

    Returns:
        objective: scalar value of objective function
    '''
    n = len(y)
    margin = 0
    obj = 0
    for i in range(n):
        if y[i] == 0:
            y[i] = -1

    for i in range(n):
        margin = -y[i]*(np.transpose(theta).dot(X[i,:]))
        obj += np.logaddexp(0, margin)
    obj = (obj/n) + l2_param*(np.transpose(theta).dot(theta))

    return obj


def fit_logistic_reg(X, y, objective_function, l2_param):
    '''
    Args:
        X: 2D numpy array of size (num_instances, num_features)
        y: 1D numpy array of size num_instances
        objective_function: function returning the value of the objective
        l2_param: regularization parameter
        
    Returns:
        optimal_theta: 1D numpy array of size num_features
    '''

    d = X.shape[1]

    #minimizing using scipy
    theta_initial = np.zeros(d)     

    objective = functools.partial(objective_function, X = X, y = y, l2_param= l2_param)      #passing partial function

    theta = minimize(objective, theta_initial)

    return theta.x

def b_predict(w, X):
    print(w.shape)
    dot = np.dot(X, w)
    return (1 / (1 + np.exp(-dot)))




def main():
    
    X_t = genfromtxt('X_train.txt', delimiter=',')
    X_v = genfromtxt('X_val.txt', delimiter=',')
    y_train = genfromtxt('y_train.txt', delimiter=',')
    y_val = genfromtxt('y_val.txt', delimiter=',')

    #Add as bias term
    n = len(y_train)    #training bias term additio 
    bias = np.zeros(n)
    for i in range(n):
        bias[i] = 1
    bias = bias.reshape((n,1))
    X_T = np.append(X_t,bias,1)

    n1 = len(y_val)     #validation bias term additio 
    bias = np.zeros(n1)
    for i in range(n1):
        bias[i] = 1
    bias = bias.reshape((n1,1))
    X_V = np.append(X_v,bias,1)



    #Standarize the data
    X_train = preprocessing.scale(X_T)
    X_val = preprocessing.scale(X_V)


    '''
    #Fitting the data
    #opt_theta = fit_logistic_reg(X_train, y_train, f_objective).x

    #print("Training loss =>", f_objective(opt_theta, X_train, y_train))
    #print("Validation loss =>", f_objective(opt_theta, X_val, y_val))

    #hyperparameter search = 0.0187
    l2_range = np.arange(0.018, 0.0197, 0.0001)
    loss = []
    for i in l2_range:
        opt_theta = fit_logistic_reg(X_train, y_train, f_objective, i).x
        temp_loss = f_objective(opt_theta, X_val, y_val, 0)
        loss.append(-n*temp_loss)

    print(loss)

    plt.plot(l2_range, loss)
    plt.ylabel('log-likelihood')
    plt.xlabel('l2 value')
    plt.show()
    '''

    #3.3.5
    
    w = fit_logistic_reg(X_train, y_train, f_objective, l2_param=0.0187)
    predicts = b_predict(w, X_val)

    bin_means, bin_edges, bin_num = binned_statistic(predicts, y_val, bins=12)

    plt.hlines(bin_means, bin_edges[:-1], bin_edges[1:])
    plt.xlabel("Bins")
    plt.ylabel("Mean of y_val")

    plt.show()
    
    






if __name__ == '__main__':
    main()













        