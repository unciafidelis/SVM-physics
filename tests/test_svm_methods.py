import os
import sys
import unittest
import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC, LinearSVC


data = DataPreparation(path="../data/", GA_selection=True)
X_train, y_train, X_test, y_test = data.dataset(sample_name="titanic", sampling=False, split_sample=0.4)


class TestLinearPrecomputed(unittest.TestCase):
    """Tests classes in common_methods"""
    @pytest.fixture(autouse=True)
    def initdir(self, tmpdir):
        print(tmpdir)
        tmpdir.chdir() # change to pytest-provided temporary directory

    def test_linear_precomputed(self):
        from common.svm_methods import LinearPrecomputed
        test_linear = LinearPrecomputed(x_input=X_train)
        matrix_ep = test_linear.explicit_calc()
        matrix_np = test_linear.numpy_calc()
        model_og = SVC(kernel="linear")
        model_ep = SVC(kernel="precomputed")
        model_np = SVC(kernel="precomputed")
        
        model_og.fit(X_train, y_train)
        model_ep.fit(matrix_ep, y_train)
        model_np.fit(matrix_np, y_train)

        y_pred_og = model_og.predict(X_test)
        acc_og = accuracy_score(y_test, y_pred_og)
        prc_og = precision_score(y_test, y_pred_og)

        matrix_ep_test = LinearPrecomputed(x_input=X_test)
        matrix_ep_test = matrix_ep_test.explicit_calc()
        y_pred_ep = model_og.predict(X_test) # matrix_ep_test)
        acc_ep = accuracy_score(y_test, y_pred_ep)
        prc_ep = precision_score(y_test, y_pred_ep)

        matrix_np_test = LinearPrecomputed(x_input=X_test)
        matrix_np_test = matrix_np_test.explicit_calc()
        y_pred_np = model_og.predict(X_test) # matrix_np_test)
        acc_np = accuracy_score(y_test, y_pred_np)
        prc_np = precision_score(y_test, y_pred_np)

        print(acc_og, prc_og, acc_ep, prc_ep, acc_np, prc_np)
        #self.assertCountEqual(ordered_list, list(X_train.columns))


if __name__ == '__main__':
    unittest.main()
