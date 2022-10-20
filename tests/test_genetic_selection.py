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

data = DataPreparation(path="../data/", GA_selection=False)
X_train, Y_train, X_test, Y_test = data.dataset(sample_name="titanic", sampling=False, split_sample=0.5)


class TestGeneticSelection(unittest.TestCase):
    """Tests classes in common_methods"""
    @pytest.fixture(autouse=True)
    def initdir(self, tmpdir):
        print(tmpdir)
        tmpdir.chdir()

    def test_genetic_selection_simple(self):
        from common.genetic_selection import GeneticSelection
        model = SVC()
        model.fit(X_train, Y_train)
        model = SVC()
        genetic = GeneticSelection(model, "deci", X_train, Y_train, X_test, Y_test,
                                   pop_size=10, chrom_len=250, n_gen=50, coef=0.5,
                                   mut_rate=0.3, score_type="auc", selec_type="tournament")
        genetic.execute()
        indexes = genetic.best_pop.flatten()
        indexes_clean = genetic.best_population()
        best_train_indexes = np.unique(genetic.best_pop.flatten())
        best_train_indexes = indexes_clean
        Y_final = Y_train[best_train_indexes]
        # print(len(Y_final[Y_final==1]), len(Y_final[Y_final==-1]), len(best_train_indexes), len(Y_final), len(Y_train), ' seleccion genetica balanceado?' )
        self.assertGreater(len(indexes), len(indexes_clean))


if __name__ == '__main__':
    unittest.main()
