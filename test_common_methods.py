import os
import sys
import unittest
import pytest

from sklearn import datsets, svm


class TestModelSaving(unittest.TestCase):
    """Tests basic functions in common_methods"""
    @pytest.fixture(autouse=True)
    def initdir(self, tmpdir):
        print(tmpdir)
        tmpdir.chdir() # change to pytest-provided temporary directory

    
    def test_save_model(self):
        from common_methods import ModelSaving
        test_saver = ModelSaving()


        test_saver.save_model()
        return None


    def test_load_model(self):
        from common_methods import ModelSaving


        return None
        
        









