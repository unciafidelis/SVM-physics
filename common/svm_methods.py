import math
import numpy as np
from sklearn.svm import SVC, LinearSVC


class LinearPrecomputed():
    """
    Class that computes by hand a linear kernel
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
    Class that computes by hand a RBF kernel
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
            norm += (v1[i] - v2[i]) * (v1[i] - v2[i])
        return math.exp(-(norm) * self.gamma)
  
    def explicit_calc(self):
        return np.array([[self.rbf_explicit(i, j) for j in self.y ] for i in self.x])

    def rbf_np(self, v1, v2):
        norm = ((v1 - v2).dot(v1 - v2))
        return math.exp(-(norm) * self.gamma)
   
    def numpy_calc(self):
        return np.array([[self.rbf_np(i, j) for j in self.y ] for i in self.x])


class LaplacePrecomputed():
    """
    Class that computes by hand a Laplace kernel
    Returns a numpy array that is the Laplace transform of the input matrices
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
            norm += abs((v1[i] - v2[i]))
        return math.exp(-(norm) * self.gamma)
  
    def explicit_calc(self):
        return np.array([[self.rbf_explicit(i, j) for j in self.y ] for i in self.x])


class PolyPrecomputed():
    """
    Class that computes by hand a polynomial kernel
    Returns a numpy array that is the gaussian transform of the input matrices
    """
    def __init__(self, x, y=None, gamma=1, deg=1, coef=0):
        self.x = np.array(x)
        self.gamma = gamma
        self.deg = deg
        self.coef = coef
        if y is None:
            self.y = np.array(x)
        else:
            self.y = np.array(y)

    def poly_explicit(self, v1, v2):
        prod = 0
        for i in range(0, len(v1)):
            prod += v1[i] * v2[i]
        prod = self.gamma * prod + self.coef
        return math.pow(prod, self.deg)
  
    def explicit_calc(self):
        return np.array([[self.poly_explicit(i, j) for j in self.y ] for i in self.x])

    def poly_np(self, v1, v2):
        prod = np.dot(v1, v2.T)
        prod = self.gamma * prod + self.coef
        return math.pow(prod, self.deg)
   
    def numpy_calc(self):
        return np.array([[self.poly_np(i, j) for j in self.y ] for i in self.x])


class KernelSum():
    """
    Class that computest the linear combination of kernels
    result = a * A + b * B + ...
    Input values are gram matrices (kernel matrices)
    """
    def __init__(self, kernels=[]):
        if len(kernels) == 0:
            print("No input kernels. Goodbye!")
            return None
        else:
            self.kernels = kernels

    def linear_combination(self):
        suma = 0
        for kernel in self.kernels:
            suma += kernel[0] * kernel[1]
        return suma


class KernelProd():
    """
    Class that computes the n-product of kernels
    result = (a*A) * (b*B) * ...
    Input values are gram matrices (kernel matrices)
    """
    def __init__(self, kernels=[]):
        if len(kernels) <= 1:
            print("Not enough kernels. Goodbye!")
            return None
        else:
            self.kernels = kernels

    def matrix_product(self):
        """
        Computes the Hadamard product of a list of kernels
        """
        prod1 = self.kernels[0][0] * self.kernels[0][1]
        prod2 = self.kernels[1][0] * self.kernels[1][1]
        result = np.multiply(prod1, prod2)
        for i in range(len(self.kernels) - 2):
            temp = self.kernels[i+2][0] * self.kernels[i+2][1]
            result = np.multiply(result, temp)
        return result


def precompute_kernel(kernel_fcn, X_train, X_test=None):

    if kernel_fcn == "rbf":
        if X_test is None:
            matrix_kernel = RBFPrecomputed(X_train)
        else:
            matrix_kernel = RBFPrecomputed(X_test, X_train)
        return matrix_kernel.explicit_calc()
    elif kernel_fcn == "pol":
        if X_test is None:
            matrix_kernel = PolyPrecomputed(X_train)
        else:
            matrix_kernel = PolyPrecomputed(X_test, X_train)
        return matrix_kernel.explicit_calc()
    elif kernel_fcn == "lap":
        if X_test is None:
            matrix_kernel = LaplacePrecomputed(X_train)
        else:
            matrix_kernel = LaplacePrecomputed(X_test, X_train)
        return matrix_kernel.explicit_calc()
    elif kernel_fcn == "lin":
        if X_test is None:
            matrix_kernel = LinearPrecomputed(X_train)
        else:
            matrix_kernel = LinearPrecomputed(X_test, X_train)
        return matrix_kernel.explicit_calc()
    elif kernel_fcn == "sum_rbf_sig":
        if X_test is None:
            kernel1 = RBFPrecomputed(X_train).explicit_calc()
            kernel2 = RBFPrecomputed(X_train).explicit_calc()
        else:
            kernel1 = RBFPrecomputed(X_test, X_train).explicit_calc()
            kernel2 = RBFPrecomputed(X_test, X_train).explicit_calc()
        kernels = []
        kernels.append((1, kernel1))
        kernels.append((1, kernel2))
        kernel_sum = KernelSum(kernels)
        return kernel_sum.linear_combination()
            
    elif kernel_fcn == "sum_rbf_pol":
        if X_test is None:
            kernel1 = RBFPrecomputed(X_train).explicit_calc()
            kernel2 = PolyPrecomputed(X_train).explicit_calc()
        else:
            kernel1 = RBFPrecomputed(X_test, X_train).explicit_calc()
            kernel2 = PolyPrecomputed(X_test, X_train).explicit_calc()
        kernels = []
        kernels.append((1, kernel1))
        kernels.append((1, kernel2))
        kernel_sum = KernelSum(kernels)
        return kernel_sum.linear_combination()

    elif kernel_fcn == "sum_rbf_lin":
        if X_test is None:
            kernel1 = RBFPrecomputed(X_train).explicit_calc()
            kernel2 = LinearPrecomputed(X_train).explicit_calc()
        else:
            kernel1 = RBFPrecomputed(X_test, X_train).explicit_calc()
            kernel2 = LinearPrecomputed(X_test, X_train).explicit_calc()
        kernels = []
        kernels.append((1, kernel1))
        kernels.append((1, kernel2))
        kernel_sum = KernelSum(kernels)
        return kernel_sum.linear_combination()
        
    elif kernel_fcn == "sum_rbf_lap":
        if X_test is None:
            kernel1 = RBFPrecomputed(X_train).explicit_calc()
            kernel2 = LaplacePrecomputed(X_train).explicit_calc()
        else:
            kernel1 = RBFPrecomputed(X_test, X_train).explicit_calc()
            kernel2 = LaplacePrecomputed(X_test, X_train).explicit_calc()
        kernels = []
        kernels.append((1, kernel1))
        kernels.append((1, kernel2))
        kernel_sum = KernelSum(kernels)
        return kernel_sum.linear_combination()

    elif kernel_fcn == "prd_rbf_sig":
        if X_test is None:
            kernel1 = RBFPrecomputed(X_train).explicit_calc()
            kernel2 = RBFPrecomputed(X_train).explicit_calc()
        else:
            kernel1 = RBFPrecomputed(X_test, X_train).explicit_calc()
            kernel2 = RBFPrecomputed(X_test, X_train).explicit_calc()
        kernels = []
        kernels.append((1, kernel1))
        kernels.append((1, kernel2))
        kernel_prd = KernelProd(kernels)
        return kernel_prod.matrix_product()
            
    elif kernel_fcn == "prd_rbf_pol":
        if X_test is None:
            kernel1 = RBFPrecomputed(X_train).explicit_calc()
            kernel2 = PolyPrecomputed(X_train).explicit_calc()
        else:
            kernel1 = RBFPrecomputed(X_test, X_train).explicit_calc()
            kernel2 = PolyPrecomputed(X_test, X_train).explicit_calc()
        kernels = []
        kernels.append((1, kernel1))
        kernels.append((1, kernel2))
        kernel_prod = KernelProd(kernels)
        return kernel_prod.matrix_product()

    elif kernel_fcn == "prd_rbf_lin":
        if X_test is None:
            kernel1 = RBFPrecomputed(X_train).explicit_calc()
            kernel2 = LinearPrecomputed(X_train).explicit_calc()
        else:
            kernel1 = RBFPrecomputed(X_test, X_train).explicit_calc()
            kernel2 = LinearPrecomputed(X_test, X_train).explicit_calc()
        kernels = []
        kernels.append((1, kernel1))
        kernels.append((1, kernel2))
        kernel_prod = KernelProd(kernels)
        return kernel_prod.matrix_product()
        
    elif kernel_fcn == "prd_rbf_lap":
        if X_test is None:
            kernel1 = RBFPrecomputed(X_train).explicit_calc()
            kernel2 = LaplacePrecomputed(X_train).explicit_calc()
        else:
            kernel1 = RBFPrecomputed(X_test, X_train).explicit_calc()
            kernel2 = LaplacePrecomputed(X_test, X_train).explicit_calc()
        kernels = []
        kernels.append((1, kernel1))
        kernels.append((1, kernel2))
        kernel_prod = KernelProd(kernels)
        return kernel_prod.matrix_product()


    print(kernel_fcn, "+++++++++++++++++++++++++++++++++++++++++++++++++++++++")

