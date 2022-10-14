import math
import numpy as np
from sklearn.svm import SVC, LinearSVC


class LinearPrecomputed():
    """
    Class that computes by hand the linear kernel
    Returns a numpy array that is the dot product of the input matrices
    """
    def __init__(self, x, y=None):
        self.x = np.array(x)
        if y is None:
            self.y = np.array(x)
        else:
            self.y = np.array(y)

    @staticmethod
    def linear_explicit(v1, v2):
        prod = 0
        for i in range(0, len(v1)):
            prod += v1[i] * v2[i]
        return prod
  
    def explicit_calc(self):
        return np.array([[self.linear_explicit(i, j) for j in self.y ] for i in self.x])

    @staticmethod
    def linear_np(v1, v2):
        return np.dot(v1, v2.T)
   
    def numpy_calc(self):
        return np.array([[self.linear_np(i, j) for j in self.y ] for i in self.x])


class RBFPrecomputed():
    """
    Class that computes by hand the RBF kernel
    Returns a numpy array that is the gaussian transform of the input matrices
    """
    def __init__(self, x, y=None, gamma=0.01):
        self.x = np.array(x)
        self.gamma = gamma
        if y is None:
            self.y = np.array(x)
        else:
            self.y = np.array(y)

    def rbf_explicit(self, v1, v2):
        norm = 0
        for i in range(0, len(v1)):
            norm += (v1[i]-v2[i]) * (v1[i]-v2[i])
        return math.exp(-(norm) * self.gamma)
  
    def explicit_calc(self):
        return np.array([[self.rbf_explicit(i, j) for j in self.y ] for i in self.x])

    def rbf_np(self, v1, v2):
        norm = ((v1-v2).dot(v1-v2))
        return math.exp(-(norm) * self.gamma)
        # return np.dot(i, j.T)
   
    def numpy_calc(self):
        return np.array([[self.rbf_np(i, j) for j in self.y ] for i in self.x])
