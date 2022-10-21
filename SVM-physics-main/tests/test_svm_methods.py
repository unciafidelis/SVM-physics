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

data = DataPreparation(path="../data/", GA_selection=False)
X_train, y_train, X_test, y_test = data.dataset(sample_name="titanic", sampling=False, split_sample=0.5)


class TestPrecomputed(unittest.TestCase):
    """Tests classes in common_methods"""
    @pytest.fixture(autouse=True)
    def initdir(self, tmpdir):        
        print(tmpdir)
        tmpdir.chdir() # change to pytest-provided temporary directory

    def test_linear_precomputed(self):
        from common.svm_methods import LinearPrecomputed
        test_linear = LinearPrecomputed(X_train)
        matrix_test = LinearPrecomputed(X_test, X_train)
        # precomputed kernel, explicit calculation
        matrix_ep = test_linear.explicit_calc()
        model_ep = SVC(kernel="precomputed")
        model_ep.fit(matrix_ep, y_train)
        matrix_test_ep = matrix_test.explicit_calc()
        y_pred_ep = model_ep.predict(matrix_test_ep)
        acc_ep = accuracy_score(y_test, y_pred_ep)
        prc_ep = precision_score(y_test, y_pred_ep)
        # precomputed kernel, numpy calculation
        matrix_np = test_linear.numpy_calc()
        model_np = SVC(kernel="precomputed")
        model_np.fit(matrix_np, y_train)
        matrix_test_np = matrix_test.numpy_calc()
        y_pred_np = model_np.predict(matrix_test_np)
        acc_np = accuracy_score(y_test, y_pred_np)
        prc_np = precision_score(y_test, y_pred_np)
        # default(original) kernel
        model_og = SVC(kernel="linear")
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
        test_rbf = RBFPrecomputed(X_train)
        matrix_test = RBFPrecomputed(X_test, X_train)
        # precomputed kernel, explicit calculation
        matrix_ep = test_rbf.explicit_calc()
        model_ep = SVC(kernel="precomputed")
        model_ep.fit(matrix_ep, y_train)
        matrix_test_ep = matrix_test.explicit_calc()
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

    def test_poly_precomputed(self):
        from common.svm_methods import PolyPrecomputed
        test_poly = PolyPrecomputed(X_train, gamma=0.5, deg=2, coef=1)
        matrix_test = PolyPrecomputed(X_test, X_train, gamma=0.5, deg=2, coef=1)
        # precomputed kernel, explicit calculation
        matrix_ep = test_poly.explicit_calc()
        model_ep = SVC(kernel="precomputed")
        model_ep.fit(matrix_ep, y_train)
        matrix_test_ep = matrix_test.explicit_calc()
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
        test_poly = PolyPrecomputed(X_train, gamma=1, deg=1, coef=0)
        matrix_test = PolyPrecomputed(X_test, X_train, gamma=1, deg=1, coef=0)
        matrix_poly = test_poly.explicit_calc()
        matrix_test = matrix_test.explicit_calc()
        kernels = []
        kernels.append((1, matrix_poly))
        kernels.append((1, matrix_poly))
        kernels_test = []
        kernels_test.append((1, matrix_test))
        kernels_test.append((1, matrix_test))
        from common.svm_methods import KernelSum
        kernel_sum = KernelSum(kernels)
        kernel_sum = kernel_sum.linear_combination()
        kernel_sum_test = KernelSum(kernels_test)
        kernel_sum_test = kernel_sum_test.linear_combination()
        model_np = SVC(kernel="precomputed")
        model_np.fit(kernel_sum, y_train)
        y_pred_np = model_np.predict(kernel_sum_test)
        acc_np = accuracy_score(y_test, y_pred_np)
        prc_np = precision_score(y_test, y_pred_np)
        model_og = SVC(kernel="poly", degree=1, gamma=2, coef0=0)
        model_og.fit(X_train, y_train)
        y_pred_og = model_og.predict(X_test)
        acc_og = accuracy_score(y_test, y_pred_og)
        prc_og = precision_score(y_test, y_pred_og)
        self.assertEqual(acc_og, acc_np)
        self.assertEqual(prc_og, prc_np)

if __name__ == '__main__':
    unittest.main()
