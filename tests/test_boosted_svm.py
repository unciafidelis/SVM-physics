import os
import sys
import unittest
import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,precision_score,auc
from sklearn.svm import SVC, LinearSVC
# framework includes
from data.data_preparation import DataPreparation
from common.common_methods import roc_curve_adaboost,plot_roc_curve

data = DataPreparation(path="../data/", GA_selection=False)
X_train, Y_train, X_test, Y_test = data.dataset(sample_name="titanic", sampling=False, split_sample=0.5)


class TestBoostedSVM(unittest.TestCase):
    """Tests classes in common_methods"""
    @pytest.fixture(autouse=True)
    def initdir(self, tmpdir):
        print(tmpdir)
        tmpdir.chdir()

    def test_boosted_svm_simple(self):
        from common.boosted_svm import BoostedSVM
        svm_boost = BoostedSVM(C=100, gammaEnd=100, myKernel='rbf', myDegree=1, myCoef0=+1,
                               Diversity=False, early_stop=True, debug=False)
        svm_boost.fit(X_train, Y_train)
        y_preda = svm_boost.predict(X_test)
        y_thresholds = svm_boost.decision_thresholds(X_test, glob_dec=True)
        TPR, FPR = roc_curve_adaboost(y_thresholds, Y_test)
        prec = precision_score(Y_test, y_preda)
        acc = accuracy_score(Y_test, y_preda)
        area = auc(FPR,TPR)
        nWeaks = len(svm_boost.alphas)
        plot_roc_curve(TPR,FPR)
        self.assertGreater(prec, 0.5)
        self.assertGreater(acc, 0.5)
        self.assertGreater(area, 0.5)
        self.assertGreater(nWeaks, 1)

    def test_boosted_svm_div(self):
        from common.boosted_svm import BoostedSVM
        svm_boost = BoostedSVM(C=100, gammaEnd=100, myKernel='rbf', myDegree=1, myCoef0=+1,
                               Diversity=True, early_stop=True, debug=False)
        svm_boost.fit(X_train, Y_train)
        y_preda = svm_boost.predict(X_test)
        y_thresholds = svm_boost.decision_thresholds(X_test, glob_dec=True)
        TPR, FPR = roc_curve_adaboost(y_thresholds, Y_test)
        prec = precision_score(Y_test, y_preda)
        acc = accuracy_score(Y_test, y_preda)
        area = auc(FPR,TPR)
        nWeaks = len(svm_boost.alphas)
        plot_roc_curve(TPR,FPR)
        self.assertGreater(prec, 0.5)
        self.assertGreater(acc, 0.5)
        self.assertGreater(area, 0.5)
        self.assertGreater(nWeaks, 1)

if __name__ == '__main__':
    unittest.main()
