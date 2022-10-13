from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import ctypes
import numpy as np
from random import sample
import pandas as pd
from sklearn.metrics import accuracy_score
import math
import os



class LinearPrecomputed():
    """
    Class that computes by hand the linear kernel
    """
    def __init__(self, x_input):
        self.x = np.array(x_input)
        self.y = np.array(x_input)

    @staticmethod
    def linear_explicit(v1, v2):
        prod = 0
        for i in range(0, len(v1)):
            prod += v1[i] * v2[i]
        return prod
  
    def explicit_calc(self):
        return np.array([[self.linear_explicit(i, j) for j in self.y ] for i in self.x])

    @staticmethod
    def linear_np(i, j):
        return np.dot(i, j.T)
   
    def numpy_calc(self):
        return np.array([[self.linear_np(i, j) for j in self.y ] for i in self.x])

