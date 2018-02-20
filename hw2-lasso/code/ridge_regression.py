"""
Ridge regression using scipy's minimize function and demonstrating the use of
sklearn's framework.

Author: David S. Rosenberg <david.davidr@gmail.com>
License: Creative Commons Attribution 4.0 International License
"""
from sklearn.base import BaseEstimator, RegressorMixin
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error, make_scorer
import pandas as pd
import random
from random import shuffle
from sklearn.metrics import mean_squared_error, make_scorer, confusion_matrix


from setup_problem import load_problem

class RidgeRegression(BaseEstimator, RegressorMixin):
    """ ridge regression"""

    def __init__(self, l2reg=1):
        if l2reg < 0:
            raise ValueError('Regularization penalty should be at least 0.')
        self.l2reg = l2reg

    def fit(self, X, y=None):
        n, num_ftrs = X.shape
        # convert y to 1-dim array, in case we're given a column vector
        y = y.reshape(-1)
        def ridge_obj(w):
            predictions = np.dot(X,w)
            residual = y - predictions
            empirical_risk = np.sum(residual**2) / n
            l2_norm_squared = np.sum(w**2)
            objective = empirical_risk + self.l2reg * l2_norm_squared
            return objective
        self.ridge_obj_ = ridge_obj

        w_0 = np.zeros(num_ftrs)
        self.w_ = minimize(ridge_obj, w_0).x
        return self

    def predict(self, X, y=None):
        try:
            getattr(self, "w_")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")
        return np.dot(X, self.w_)

    def score(self, X, y):
        # Average square error
        try:
            getattr(self, "w_")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")
        residuals = self.predict(X) - y
        return np.dot(residuals, residuals)/len(y)



def compare_our_ridge_with_sklearn(X_train, y_train, l2_reg=1):
    # First run sklearn ridge regression and extract the coefficients
    from sklearn.linear_model import Ridge
    # Fit with sklearn -- need to multiply l2_reg by sample size, since their
    # objective function has the total square loss, rather than average square
    # loss.
    n = X_train.shape[0]
    sklearn_ridge = Ridge(alpha=n*l2_reg, fit_intercept=False, normalize=False)
    sklearn_ridge.fit(X_train, y_train)
    sklearn_ridge_coefs = sklearn_ridge.coef_

    # Now run our ridge regression and compare the coefficients to sklearn's
    ridge_regression_estimator = RidgeRegression(l2reg=l2_reg)
    ridge_regression_estimator.fit(X_train, y_train)
    our_coefs = ridge_regression_estimator.w_

    print("Hoping this is very close to 0:{}".format(np.sum((our_coefs - sklearn_ridge_coefs)**2)))

def do_grid_search_ridge(X_train, y_train, X_val, y_val):
    # Now let's use sklearn to help us do hyperparameter tuning
    # GridSearchCv.fit by default splits the data into training and
    # validation itself; we want to use our own splits, so we need to stack our
    # training and validation sets together, and supply an index
    # (validation_fold) to specify which entries are train and which are
    # validation.
    X_train_val = np.vstack((X_train, X_val))
    y_train_val = np.concatenate((y_train, y_val))
    val_fold = [-1]*len(X_train) + [0]*len(X_val) #0 corresponds to validation

    # Now we set up and do the grid search over l2reg. The np.concatenate
    # command illustrates my search for the best hyperparameter. In each line,
    # I'm zooming in to a particular hyperparameter range that showed promise
    # in the previous grid. This approach works reasonably well when
    # performance is convex as a function of the hyperparameter, which it seems
    # to be here.
    param_grid = [{'l2reg':np.unique(np.concatenate((10.**np.arange(-8,1,1),
                                           np.arange(1,5,.3)
                                             ))) }]


    ridge_regression_estimator = RidgeRegression()
    grid = GridSearchCV(ridge_regression_estimator,
                        param_grid,
                        return_train_score=True,
                        cv = PredefinedSplit(test_fold=val_fold),
                        refit = True,
                        scoring = make_scorer(mean_squared_error,
                                              greater_is_better = False))
    grid.fit(X_train_val, y_train_val)

    df = pd.DataFrame(grid.cv_results_)
    # Flip sign of score back, because GridSearchCV likes to maximize,
    # so it flips the sign of the score if "greater_is_better=FALSE"
    df['mean_test_score'] = -df['mean_test_score']
    df['mean_train_score'] = -df['mean_train_score']
    cols_to_keep = ["param_l2reg", "mean_test_score","mean_train_score"]
    df_toshow = df[cols_to_keep].fillna('-')
    df_toshow = df_toshow.sort_values(by=["param_l2reg"])
    return grid, df_toshow

def compare_parameter_vectors(pred_fns):
    # Assumes pred_fns is a list of dicts, and each dict has a "name" key and a
    # "coefs" key
    fig, axs = plt.subplots(len(pred_fns),1, sharex=True)
    num_ftrs = len(pred_fns[0]["coefs"])
    for i in range(len(pred_fns)):
        title = pred_fns[i]["name"]
        coef_vals = pred_fns[i]["coefs"]
        if i >0:
            for j in range(len(coef_vals)):
                if abs(coef_vals[j]) < 0:
                    coef_vals[j] = 0
        axs[i].bar(range(num_ftrs), coef_vals)
        axs[i].set_xlabel('Feature Index')
        axs[i].set_ylabel('Parameter Value')
        axs[i].set_title(title)

    fig.subplots_adjust(hspace=0.3)
    return fig

def plot_prediction_functions(x, pred_fns, x_train, y_train, legend_loc="best"):
    # Assumes pred_fns is a list of dicts, and each dict has a "name" key and a
    # "preds" key. The value corresponding to the "preds" key is an array of
    # predictions corresponding to the input vector x. x_train and y_train are
    # the input and output values for the training data
    fig, ax = plt.subplots()
    ax.set_xlabel('Input Space: [0,1)')
    ax.set_ylabel('Action/Outcome Space')
    ax.set_title("Prediction Functions")
    plt.scatter(x_train, y_train, label='Training data')
    for i in range(len(pred_fns)):
        ax.plot(x, pred_fns[i]["preds"], label=pred_fns[i]["name"])
    legend = ax.legend(loc=legend_loc, shadow=True)
    return fig

##Binarizer

def binarize(arr, threshold):
    tempArr = np.zeros(arr.shape[0])
    for i in range(len(arr)):
        if abs(arr[i]) > threshold:
            tempArr[i] = 1
    return tempArr        
            


### Coordinate Descent:

def soft(a, c, l1reg):
    if not a == 0:
        sgn = np.sign(c/a)
        remain = max(0, abs(c/a) - (l1reg/a))
        final = sgn*remain
    elif a == 0:
        final = 0
    return final

def coordinate_descent(X, y, l1reg):
    # warm starting 
    
    n, num_ftrs = X.shape
    '''
    one = np.identity(num_ftrs) 
    w0 = ( (X.transpose()).dot(X) + l1reg*one)
    inverse = np.linalg.inv(w0)
    w1 = inverse.dot(X.transpose())
    w = w1.dot(y)
    '''
    w = np.zeros(num_ftrs)

    for i in range(1000):
        for j in range(num_ftrs):
            Xj = X[:,j]
            a = 2*np.dot(Xj, Xj.transpose())
            in_val = y - X.dot(w) + w[j]*Xj
            c = 2*Xj.dot(in_val)
            w[j] = soft(a,c,l1reg) 
    return w

def random_coordinate_descent(X, y, l1reg):
    n, num_ftrs = X.shape 
    #w = np.zeros(num_ftrs)

    #warming it up:
    one = np.identity(num_ftrs) 
    w0 = ( (X.transpose()).dot(X) + l1reg*one)
    inverse = np.linalg.inv(w0)
    w1 = inverse.dot(X.transpose())
    w = w1.dot(y)

    for i in range(1000):
        rand_lst = np.arange(num_ftrs)
        np.random.shuffle(rand_lst)
        for j in rand_lst:
            Xj = X[:,j]
            a = 2*np.dot(Xj, Xj.transpose())
            in_val = y - X.dot(w) + w[j]*Xj
            c = 2*Xj.dot(in_val)
            w[j] = soft(a,c,l1reg) 
    return w


def shoot_predict(X, w):
    return X.dot(w)

def compute_square_loss(X, y, theta):

    loss = 0 #initialize the square_loss
    m = X.shape[0]      #no. of training cases

    normal = (X.dot(theta) - y)
    transpose = ((X.dot(theta) - y).transpose())

    loss = (transpose.dot(normal))/m       #Square loss in Matrix form
    return(loss)

def do_grid_search_lasso(X_train, y_train, X_val, y_val):

    l1_val = []
    cost = []
    start = 0.5
    for l1reg in range (15):
        shooting_regression_estimator = random_coordinate_descent(X_train, y_train, start) 
        pred_cost = compute_square_loss(X_val, y_val, shooting_regression_estimator)
        l1_val.append(start)
        cost.append(pred_cost)
        start = start + 0.2
    #print table
    print(" sl1_val", "                       ", "cost")
    for i in range(10):
        print(np.around(l1_val[i], decimals=2), "                          ", np.around(cost[i], decimals=4))
    
    plt.plot(l1_val, cost)
    plt.xlabel("L1 Values")
    plt.ylabel("Cost")
    plt.show()

    return



def main():
    lasso_data_fname = "lasso_data.pickle"
    x_train, y_train, x_val, y_val, target_fn, coefs_true, featurize = load_problem(lasso_data_fname)

    # Generate features
    X_train = featurize(x_train)
    X_val = featurize(x_val)


    ''' 
    #Visualize training data
    fig, ax = plt.subplots()
    ax.imshow(X_train)
    ax.set_title("Design Matrix: Color is Feature Value")
    ax.set_xlabel("Feature Index")
    ax.set_ylabel("Example Number")
    #fig.show()

    # Compare our RidgeRegression to sklearn's.
    #compare_our_ridge_with_sklearn(X_train, y_train, l2_reg = 1.5)

    # Do hyperparameter tuning with our ridge regression
    grid, results = do_grid_search_ridge(X_train, y_train, X_val, y_val)
    #print(results)

    # Plot validation performance vs regularization parameter
    fig, ax = plt.subplots()
    #ax.loglog(results["param_l2reg"], results["mean_test_score"])
    ax.semilogx(results["param_l2reg"], results["mean_test_score"])
    ax.grid()
    ax.set_title("Validation Performance vs L2 Regularization")
    ax.set_xlabel("L2-Penalty Regularization Parameter")
    ax.set_ylabel("Mean Squared Error")
    #fig.show()

    '''
    
   
    #Applying coordinate descent:
    pred_fns = []
    l1reg = 1.5
    x = np.sort(np.concatenate([np.arange(0,1,.001), x_train]))
    X = featurize(x)
    name = "Target Parameter Values (i.e. Bayes Optimal)"
    pred_fns.append({"name":name, "coefs":coefs_true, "preds": target_fn(x) })
    
    name = "Shooting algorithm with L1Reg = 1.5"

    #shooting_regression_estimator = random_coordinate_descent(X_train, y_train, l1reg)
    shooting_regression_estimator = coordinate_descent(X_train, y_train, l1reg)
    shooting_regression_prediction = shoot_predict(X, shooting_regression_estimator)

    #print(type(shooting_regression_estimator))
    #print(shooting_regression_prediction.shape)
    pred_fns.append({"name":name,
                     "coefs":shooting_regression_estimator,
                     "preds": shooting_regression_prediction })

    #f = plot_prediction_functions(x, pred_fns, x_train, y_train, legend_loc="best")
    #plt.show()

    #f = compare_parameter_vectors(pred_fns)
    #plt.show()

    do_grid_search_lasso(X_train, y_train, X_val, y_val)

    
  
    '''

    # Let's plot prediction functions and compare coefficients for several fits
    # and the target function.
    pred_fns = []
    x = np.sort(np.concatenate([np.arange(0,1,.001), x_train]))
    name = "Target Parameter Values (i.e. Bayes Optimal)"
    pred_fns.append({"name":name, "coefs":coefs_true, "preds": target_fn(x) })

    l2regs = [0,grid.best_params_['l2reg']]
    X = featurize(x)

    
    ridge_regression_estimator = RidgeRegression(l2reg =0.01)
    ridge_regression_estimator.fit(X_train, y_train)
    name = "Ridge with L2Reg=" #+str(l2reg)
    pred_fns.append({"name":name,
                     "coefs":ridge_regression_estimator.w_,
                     "preds": ridge_regression_estimator.predict(X) })


     #Confusion matrix
    #conf_matrix = confusion_matrix(binarize(coefs_true, 0), binarize(coefs_ridge, 10*-6), labels=None, sample_weight=None)


    coefs_ridge = ridge_regression_estimator.w_
    for i in (10**-6, 10**-5, 10**-4, 10**-3, 10**-2, 10**-1):
        print("Confusion matrix with threshold " + str(i) +" is ", confusion_matrix(binarize(coefs_true, 0), binarize(coefs_ridge, i)))
    
    
    f = plot_prediction_functions(x, pred_fns, x_train, y_train, legend_loc="best")
    #plt.show()

    f = compare_parameter_vectors(pred_fns)
    #plt.show()
    '''
 

    

if __name__ == '__main__':
  main()
