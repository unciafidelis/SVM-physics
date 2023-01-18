import os
import sys
import unittest
import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,precision_score
from sklearn.svm import SVC, LinearSVC
# framework includes
from data.data_preparation import DataPreparation

data = DataPreparation(path=".", GA_selection=False)
X_train, y_train, X_test, y_test = data.dataset(sample_name="titanic", sampling=False, split_sample=0.5)


class TestPrecomputed(unittest.TestCase):
    """
    Tests classes in svm_methods
    """
    @pytest.fixture(autouse=True)
    def initdir(self, tmpdir):        
        print(tmpdir)
        tmpdir.chdir() # change to pytest-provided temporary directory

    def test_linear_precomputed(self):
        from common.svm_methods import LinearPrecomputed
        test_linear = LinearPrecomputed([X_train])
        matrix_test = LinearPrecomputed([X_test, X_train])
        # precomputed kernel, explicit calculation
        matrix_ep = test_linear.compute()
        model_ep = SVC(kernel="precomputed", C=100, gamma=0.01)
        model_ep.fit(matrix_ep, y_train)
        matrix_test_ep = matrix_test.compute()
        y_pred_ep = model_ep.predict(matrix_test_ep)
        acc_ep = accuracy_score(y_test, y_pred_ep)
        prc_ep = precision_score(y_test, y_pred_ep)
        # precomputed kernel, numpy calculation
        matrix_np = test_linear.numpy_calc()
        model_np = SVC(kernel="precomputed", C=100, gamma=0.01)
        model_np.fit(matrix_np, y_train)
        matrix_test_np = matrix_test.numpy_calc()
        y_pred_np = model_np.predict(matrix_test_np)
        acc_np = accuracy_score(y_test, y_pred_np)
        prc_np = precision_score(y_test, y_pred_np)
        # default(original) kernel
        model_og = SVC(kernel="linear", C=100, gamma=0.01)
        model_og.fit(X_train, y_train)
        y_pred_og = model_og.predict(X_test)
        acc_og = accuracy_score(y_test, y_pred_og)
        prc_og = precision_score(y_test, y_pred_og)
        self.assertEqual(acc_og, acc_ep)
        self.assertEqual(prc_og, prc_ep)
        self.assertEqual(acc_og, acc_np)
        self.assertEqual(prc_og, prc_np)
        
    def test_rbf_precomputed(self):
        from common.svm_methods import RBFPrecomputed
        test_rbf = RBFPrecomputed([X_train])
        matrix_test = RBFPrecomputed([X_test, X_train])
        # precomputed kernel, explicit calculation
        matrix_ep = test_rbf.compute()
        model_ep = SVC(kernel="precomputed")
        model_ep.fit(matrix_ep, y_train)
        matrix_test_ep = matrix_test.compute()
        y_pred_ep = model_ep.predict(matrix_test_ep)
        acc_ep = accuracy_score(y_test, y_pred_ep)
        prc_ep = precision_score(y_test, y_pred_ep)
        # precomputed kernel, numpy calculation
        matrix_np = test_rbf.numpy_calc()
        model_np = SVC(kernel="precomputed")
        model_np.fit(matrix_np, y_train)
        matrix_test_np = matrix_test.numpy_calc()
        y_pred_np = model_np.predict(matrix_test_np)
        acc_np = accuracy_score(y_test, y_pred_np)
        prc_np = precision_score(y_test, y_pred_np)
        # default(original) kernel
        model_og = SVC(kernel="rbf", gamma=0.01)
        model_og.fit(X_train, y_train)
        y_pred_og = model_og.predict(X_test)
        acc_og = accuracy_score(y_test, y_pred_og)
        prc_og = precision_score(y_test, y_pred_og)
        self.assertEqual(acc_og, acc_ep)
        self.assertEqual(prc_og, prc_ep)
        self.assertEqual(acc_og, acc_np)
        self.assertEqual(prc_og, prc_np)

    def test_sigmoid_precomputed(self):
        from common.svm_methods import SigmoidPrecomputed
        test_sig = SigmoidPrecomputed([X_train], gamma=0.1)
        matrix_test = SigmoidPrecomputed([X_test, X_train], gamma=0.1)
        # precomputed kernel, explicit calculation
        matrix_ep = test_sig.compute()
        model_ep = SVC(kernel="precomputed")
        model_ep.fit(matrix_ep, y_train)
        matrix_test_ep = matrix_test.compute()
        y_pred_ep = model_ep.predict(matrix_test_ep)
        acc_ep = accuracy_score(y_test, y_pred_ep)
        prc_ep = precision_score(y_test, y_pred_ep)
        # default(original) kernel
        model_og = SVC(kernel="sigmoid", gamma=0.1)
        model_og.fit(X_train, y_train)
        y_pred_og = model_og.predict(X_test)
        acc_og = accuracy_score(y_test, y_pred_og)
        prc_og = precision_score(y_test, y_pred_og)
        self.assertEqual(acc_og, acc_ep)
        self.assertEqual(prc_og, prc_ep)

    def test_laplace_precomputed(self):
        from common.svm_methods import LaplacePrecomputed
        test_lap = LaplacePrecomputed([X_train])
        matrix_test = LaplacePrecomputed([X_test, X_train])
        # precomputed kernel, explicit calculation
        matrix_ep = test_lap.compute()
        model_ep = SVC(kernel="precomputed")
        model_ep.fit(matrix_ep, y_train)
        matrix_test_ep = matrix_test.compute()
        y_pred_ep = model_ep.predict(matrix_test_ep)
        acc_ep = accuracy_score(y_test, y_pred_ep)
        prc_ep = precision_score(y_test, y_pred_ep)
        # calculated with sklearn
        from sklearn.metrics.pairwise import laplacian_kernel
        model_og = SVC(kernel="precomputed")
        matrix_og = laplacian_kernel(X_train, gamma=0.01)
        model_og.fit(matrix_og, y_train)
        matrix_test_og = laplacian_kernel(X_test, X_train, gamma=0.01)
        y_pred_og = model_og.predict(matrix_test_og)
        acc_og = accuracy_score(y_test, y_pred_og)
        prc_og = precision_score(y_test, y_pred_og)
        self.assertEqual(acc_og, acc_ep)
        self.assertEqual(prc_og, prc_ep)

    def test_poly_precomputed(self):
        from common.svm_methods import PolyPrecomputed
        test_poly = PolyPrecomputed([X_train], gamma=0.5, deg=2, coef=1)
        matrix_test = PolyPrecomputed([X_test, X_train], gamma=0.5, deg=2, coef=1)
        # precomputed kernel, explicit calculation
        matrix_ep = test_poly.compute()
        model_ep = SVC(kernel="precomputed")
        model_ep.fit(matrix_ep, y_train)
        matrix_test_ep = matrix_test.compute()
        y_pred_ep = model_ep.predict(matrix_test_ep)
        acc_ep = accuracy_score(y_test, y_pred_ep)
        prc_ep = precision_score(y_test, y_pred_ep)
        # precomputed kernel, numpy calculation
        matrix_np = test_poly.numpy_calc()
        model_np = SVC(kernel="precomputed")
        model_np.fit(matrix_np, y_train)
        matrix_test_np = matrix_test.numpy_calc()
        y_pred_np = model_np.predict(matrix_test_np)
        acc_np = accuracy_score(y_test, y_pred_np)
        prc_np = precision_score(y_test, y_pred_np)
        # default(original) kernel
        model_og = SVC(kernel="poly", degree=2, gamma=0.5, coef0=1)
        model_og.fit(X_train, y_train)
        y_pred_og = model_og.predict(X_test)
        acc_og = accuracy_score(y_test, y_pred_og)
        prc_og = precision_score(y_test, y_pred_og)
        self.assertEqual(acc_og, acc_ep)
        self.assertEqual(prc_og, prc_ep)
        self.assertEqual(acc_og, acc_np)
        self.assertEqual(prc_og, prc_np)

    def test_kernel_sum(self):
        from common.svm_methods import PolyPrecomputed
        kernel_comp = PolyPrecomputed([X_train], gamma=1, deg=1, coef=0)
        kernel_test_comp = PolyPrecomputed([X_test, X_train], gamma=1, deg=1, coef=0)
        kernel_comp = kernel_comp.compute()
        kernel_test_comp = kernel_test_comp.compute()
        kernels = []
        kernels.append((1, kernel_comp))
        kernels.append((1, kernel_comp))
        kernels_test_comp = []
        kernels_test_comp.append((1, kernel_test_comp))
        kernels_test_comp.append((1, kernel_test_comp))
        from common.svm_methods import KernelSum
        kernel_sum = KernelSum(kernels)
        kernel_sum = kernel_sum.compute()
        kernel_sum_test = KernelSum(kernels_test_comp)
        kernel_sum_test = kernel_sum_test.compute()
        model_sum = SVC(kernel="precomputed")
        model_sum.fit(kernel_sum, y_train)
        y_pred_sum = model_sum.predict(kernel_sum_test)
        acc_sum = accuracy_score(y_test, y_pred_sum)
        prc_sum = precision_score(y_test, y_pred_sum)
        model_og = SVC(kernel="poly", degree=1, gamma=2, coef0=0)
        model_og.fit(X_train, y_train)
        y_pred_og = model_og.predict(X_test)
        acc_og = accuracy_score(y_test, y_pred_og)
        prc_og = precision_score(y_test, y_pred_og)
        self.assertEqual(acc_og, acc_sum)
        self.assertEqual(prc_og, prc_sum)

    def test_kernel_prod(self):
        from common.svm_methods import PolyPrecomputed
        kernel_comp = PolyPrecomputed([X_train], gamma=1, deg=1, coef=0)
        kernel_test_comp = PolyPrecomputed([X_test, X_train], gamma=1, deg=1, coef=0)
        kernel_comp = kernel_comp.compute()
        kernel_test_comp = kernel_test_comp.compute()
        ones_train = np.ones((len(X_train), len(X_train)))
        ones_test  = np.ones((len(X_test), len(X_train)))
        kernels = []
        kernels.append((1, kernel_comp))
        kernels.append((1, kernel_comp))
        kernels.append((1, kernel_comp))
        kernels.append((1, ones_train))
        kernels.append((1, ones_train))
        kernels.append((1, ones_train))
        kernels_test_comp = []
        kernels_test_comp.append((1, kernel_test_comp))
        kernels_test_comp.append((1, kernel_test_comp))
        kernels_test_comp.append((1, kernel_test_comp))
        kernels_test_comp.append((1, ones_test))
        kernels_test_comp.append((1, ones_test))
        kernels_test_comp.append((1, ones_test))
        from common.svm_methods import KernelProd
        kernel_multiplication = KernelProd(kernels)
        kernel_multiplication = kernel_multiplication.compute()
        kernel_multiplication_test = KernelProd(kernels_test_comp)
        kernel_multiplication_test = kernel_multiplication_test.compute()
        model_prod = SVC(kernel="precomputed")
        model_prod.fit(kernel_multiplication, y_train)
        y_predict_prod = model_prod.predict(kernel_multiplication_test)
        acc_prod = accuracy_score(y_test, y_predict_prod)
        prc_prod = precision_score(y_test, y_predict_prod)
        model_og = SVC(kernel="poly", degree=3, gamma=1, coef0=0)
        model_og.fit(X_train, y_train)
        y_pred_og = model_og.predict(X_test)
        acc_og = accuracy_score(y_test, y_pred_og)
        prc_og= precision_score(y_test, y_pred_og)
        self.assertEqual(acc_og, acc_prod)
        self.assertEqual(prc_og, prc_prod)


if __name__ == '__main__':
    unittest.main()
