import numpy as np
import matplotlib.pyplot as plt
import sklearn
import scipy.spatial
import functools
import math

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




'''
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









