import os
import sys
import unittest
import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets,tree


data_iris = datasets.load_iris()
data_set = pd.DataFrame(data=data_iris.data, columns=data_iris.feature_names)
data_set['species'] = pd.Categorical.from_codes(data_iris.target, data_iris.target_names)
species_mapping = {"setosa": 1, "versicolour": 1, "virginica": -1}
data_set["species"] = data_set["species"].map(species_mapping)
data_set["species"] = data_set["species"].fillna(+1)
Y = data_set["species"]
X = data_set.drop("species", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.6, random_state=0)
variables = list(X_train.columns)
model = tree.DecisionTreeClassifier() # svm.SVC(kernel="sigmoid", gamma=1)


class TestModelStoring(unittest.TestCase):
    """Tests classes in common_methods"""
    @pytest.fixture(autouse=True)
    def initdir(self, tmpdir):
        print(tmpdir)
        tmpdir.chdir() # change to pytest-provided temporary directory
    
    def test_store_model(self):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc_og = accuracy_score(y_test, y_pred)
        prc_og = precision_score(y_test, y_pred)
        from common.common_methods import ModelStoring
        test_store = ModelStoring(file_name="./files/test_model.pkl")
        test_store.save_model(model=model)
        model_pickled = test_store.load_model()
        y_pred = model_pickled.predict(X_test)
        acc_pk = accuracy_score(y_test, y_pred)
        prc_pk = precision_score(y_test, y_pred)
        # print(acc_og, acc_pk, prc_og, prc_pk)
        self.assertEqual(acc_og, acc_pk)
        self.assertEqual(prc_og, prc_pk)


class TestVariableImportance(unittest.TestCase):
    """Tests classes in common.common_methods"""
    @pytest.fixture(autouse=True)
    def initdir(self, tmpdir):
        print(tmpdir)
        tmpdir.chdir() # change to pytest-provided temporary directory
    
    def test_one_variable_out(self):
        from common.common_methods import VariableImportance
        test_var_imp = VariableImportance(model=model, X_train=X_train, Y_train=y_train,
                                          X_test=X_test, Y_test=y_test, roc_area="prob")
        ordered_variables = test_var_imp.one_variable_out()
        print(ordered_variables, "one out")
        ordered_list = []
        for first, second in ordered_variables:
            ordered_list.append(first)
        self.assertCountEqual(ordered_list, list(X_train.columns))

    def test_recursive_one_variable_out(self):
        from common.common_methods import VariableImportance
        test_var_imp = VariableImportance(model=model, X_train=X_train, Y_train=y_train,
                                          X_test=X_test, Y_test=y_test, roc_area="prob")
        ordered_variables_recursive = test_var_imp.recursive_one_variable_out()
        print(ordered_variables_recursive, "recursive one out")
        ordered_list = []
        for first, second in ordered_variables_recursive:
            ordered_list.append(first)
        self.assertCountEqual(ordered_list, list(X_train.columns))

    def test_permutation_feature(self):
        from common.common_methods import VariableImportance
        test_var_imp = VariableImportance(model=model, X_train=X_train, Y_train=y_train,
                                          X_test=X_test, Y_test=y_test, roc_area="prob")
        ordered_permutation = test_var_imp.permutation_feature()
        print(ordered_permutation, "permutation")
        ordered_list = []
        for first, second in ordered_permutation:
            ordered_list.append(first)
        self.assertCountEqual(ordered_list, list(X_train.columns))

    def test_shapley_values(self):
        import warnings
        warnings.simplefilter("ignore") # safe to ignore numpy warnings for now!
        from common.common_methods import VariableImportance
        test_var_imp = VariableImportance(model=model, X_train=X_train, Y_train=y_train,
                                          X_test=X_test, Y_test=y_test, roc_area="prob")

        #ordered_shapley = test_var_imp.shapley_values()
        test_var_imp.shapley_values()

        # print(ordered_permutation, "permutation")
        # ordered_list = []
        # for first, second in ordered_permutation:
        #     ordered_list.append(first)
        # self.assertCountEqual(ordered_list, list(X_train.columns))


if __name__ == '__main__':
    unittest.main()
