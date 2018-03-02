import numpy as np
import matplotlib.pyplot as plt
import sklearn
import scipy.spatial
import functools
import math
from numpy.linalg import inv
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.model_selection import GridSearchCV,PredefinedSplit
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error,make_scorer
import pandas as pd
#import qgrid 




def linear_kernel(X1, X2):
    return np.dot(X1,np.transpose(X2))

def RBF_kernel(X1,X2,sigma):
    n1 = X1.shape[0]
    n2 = X2.shape[0]
    d = X1.shape[1]
    M= np.zeros((n1,n2))
    for i in range(n1):
        for j in range(n2):
            diff = X1[i,:] - X2[j,:]
            norm = diff.dot(diff.transpose())
            M[i,j] = math.exp(-norm/(2*(sigma**2)))
    return M

def polynomial_kernel(X1, X2, offset, degree):
    n1 = X1.shape[0]
    n2 = X2.shape[0]
    d = X1.shape[1]
    M= np.zeros((n1,n2))
    for i in range(n1):
        for j in range(n2):
            norm = X1[i,:].dot(X2[j,:].transpose())
            M[i,j] = (offset + norm)**degree
    return M


# PLot kernel machine functions

plot_step = .01
xpts = np.arange(-6.0, 6, plot_step).reshape(-1,1)
prototypes = np.array([-4,-1,0,2]).reshape(-1,1)



class Kernel_Machine(object):
    def __init__(self, kernel, prototype_points, weights):
        """
        Args:
            kernel(X1,X2) - a function return the cross-kernel matrix between rows of X1 and rows of X2 for kernel k
            prototype_points - an Rxd matrix with rows mu_1,...,mu_R
            weights - a vector of length R with entries w_1,...,w_R
        """

        self.kernel = kernel
        self.prototype_points = prototype_points
        self.weights = weights
        
    def predict(self, X):
        """
        Evaluates the kernel machine on the points given by the rows of X
        Args:
            X - an nxd matrix with inputs x_1,...,x_n in the rows
        Returns:
            Vector of kernel machine evaluations on the n points in X.  Specifically, jth entry of return vector is
                Sum_{i=1}^R w_i k(x_j, mu_i)
        """
        # TODO
        
        n = X.shape[0]
        d = X.shape[1]
        r = self.prototype_points.shape[0]
        fx = np.zeros(n)
        #change sigma here
        k = self.kernel(X, self.prototype_points)
        #print(k[0,0])
        for j in range(n):
            val = 0
            for i in range(r):
            	val = val + self.weights[i]*k[j,i]
            fx[j] = val
        return fx


#Kernel Ridge Regression:

def train_kernel_ridge_regression(X, y, kernel, l2reg):
    # TODO
    k = kernel(X,X)
    n = k.shape[0]
    identity = np.identity(n)
    inv = np.linalg.inv(l2reg*identity + k)
    alpha = inv.dot(y)
    return Kernel_Machine(kernel, X, alpha)



#prepping data
data_train,data_test = np.loadtxt("krr-train.txt"),np.loadtxt("krr-test.txt")
x_train, y_train = data_train[:,0].reshape(-1,1),data_train[:,1].reshape(-1,1)
x_test, y_test = data_test[:,0].reshape(-1,1),data_test[:,1].reshape(-1,1)


#------

class KernelRidgeRegression(BaseEstimator, RegressorMixin):  
    """sklearn wrapper for our kernel ridge regression"""
     
    def __init__(self, kernel="RBF", sigma=1, degree=2, offset=1, l2reg=1):        
        self.kernel = kernel
        self.sigma = sigma
        self.degree = degree
        self.offset = offset
        self.l2reg = l2reg 

    def fit(self, X, y=None):
        """
        This should fit classifier. All the "work" should be done here.
        """
        if (self.kernel == "linear"):
            self.k = linear_kernel
        elif (self.kernel == "RBF"):
            self.k = functools.partial(RBF_kernel, sigma=self.sigma)
        elif (self.kernel == "polynomial"):
            self.k = functools.partial(polynomial_kernel, offset=self.offset, degree=self.degree)
        else:
            raise ValueError('Unrecognized kernel type requested.')
        
        self.kernel_machine_ = train_kernel_ridge_regression(X, y, self.k, self.l2reg)

        return self

    def predict(self, X, y=None):
        try:
            getattr(self, "kernel_machine_")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")

        return(self.kernel_machine_.predict(X))

    def score(self, X, y=None):
        # get the average square error
        return((self.predict(X)-y).mean()) 


## Plot the best polynomial and RBF fits you found
plot_step = .01
xpts = np.arange(-.5 , 1.5, plot_step).reshape(-1,1)
plt.plot(x_train,y_train,'o')
#Plot best polynomial fit
offset= 1.1
degree = 5
l2reg = 0.003
k = functools.partial(polynomial_kernel, offset=offset, degree=degree)
f = train_kernel_ridge_regression(x_train, y_train, k, l2reg=l2reg)
label = "Offset="+str(offset)+",Degree="+str(degree)+",L2Reg="+str(l2reg)
plt.plot(xpts, f.predict(xpts), label=label)
#Plot best RBF fit
sigma = 0.08
l2reg= 0.013
k = functools.partial(RBF_kernel, sigma=sigma)
f = train_kernel_ridge_regression(x_train, y_train, k, l2reg=l2reg)
label = "Sigma="+str(sigma)+",L2Reg="+str(l2reg)
plt.plot(xpts, f.predict(xpts), label=label)
plt.legend(loc = 'best')
plt.ylim(-1,1.75)
plt.show()




'''

#6.3.3
#change sigma here
kernel = RBF_kernel(x_train,x_train, 0.02)
l2reg = 0.0001
reg = train_kernel_ridge_regression(x_train, y_train, kernel, l2reg)


#predicting the curve

xpts = np.arange(0, 1, 0.01).reshape(-1,1)
pred = reg.predict(xpts)

plt.scatter(x_train, y_train, label = "training points")
plt.plot(xpts,pred, label = "prediction curve", color = "red")
plt.legend(loc = 'best')
plt.show()



#6.3.d
x = np.matrix('-1; 0; 1')
w = np.matrix('1; -1; 1')
k = functools.partial(RBF_kernel, sigma =1)
RBF_obj = Kernel_Machine(k, x, w)

plot_step = .01
xpts = np.arange(-6.0, 6, plot_step).reshape(-1,1)
ypts = RBF_obj.predict(xpts) 

label = "RBF"
plt.plot(xpts, ypts, label=label)
plt.legend(loc = 'best')
plt.show() 






# RBF kernel
y = RBF_kernel(prototypes, xpts, 1)
for i in range(len(prototypes)):
    label = "RBF@"+str(prototypes[i,:])
    plt.plot(xpts, y[i,:], label=label)
plt.legend(loc = 'best')
plt.show()



# Poly kernel
y = polynomial_kernel(prototypes, xpts, 1, 3)
for i in range(len(prototypes)):
    label = "Poly@"+str(prototypes[i,:])
    plt.plot(xpts, y[i,:], label=label)
plt.legend(loc = 'best')
plt.show() 




# Linear kernel
y = linear_kernel(prototypes, xpts)
for i in range(len(prototypes)):
    label = "Linear@"+str(prototypes[i,:])
    plt.plot(xpts, y[i,:], label=label)
plt.legend(loc = 'best')
plt.show() 




'''









