import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
import math

def logistic5(x, A, B, C, D, E):
    """5PL lgoistic equation."""
    #print(x, A, B, C, D, E)
    return A * (0.5 - 1.0 / (1 + np.exp(B*(x-C)))) + D * x + E

def residuals(p, y, x):
    """Deviations of data from fitted 5PL curve"""
    A,B,C,D,E = p
    err = y-logistic5(x, A, B, C, D, E)
    return err

def peval(x, p):
    """Evaluated value at x with current parameters."""
    A,B,C,D,E = p
    return logistic5(x, A, B, C, D, E)

def logistic5_regression(pred, gt):
    """
    list 
    """
    # convert to numpy array 
    pred = np.array(pred)
    gt = np.array(gt)
    print(pred.shape, gt.shape)
    # Initial guess for parameters
    p0 = [0, 1, 1, 1, 0]

    # Fit equation using least squares optimization
    plsq = leastsq(residuals, p0, args=(gt, pred))    

    # fit value 
    res = peval(pred, plsq[0])
    return res.tolist() 

# Make up some data for fitting and add noise
# In practice, y_meas would be read in from a file
# x = np.linspace(0,20,20)
# A,B,C,D = 0.5,2.5,8,7.3
# y_true = logistic5(x, A, B, C, D)
# y_meas = y_true + 0.2*npr.randn(len(x))

# # Initial guess for parameters
# p0 = [0, 1, 1, 1, 1]

# # Fit equation using least squares optimization
# plsq = leastsq(residuals, p0, args=(y_meas, x))

# # Plot results
# plt.plot(x,peval(x,plsq[0]),x,y_meas,'o',x,y_true)
# plt.title('Least-squares 4PL fit to noisy data')
# plt.legend(['Fit', 'Noisy', 'True'], loc='upper left')
# for i, (param, actual, est) in enumerate(zip('ABCD', [A,B,C,D], plsq[0])):
#     plt.text(10, 3-i*0.5, '%s = %.2f, est(%s) = %.2f' % (param, actual, param, est))
# plt.savefig('logistic.png')