import os
import sys
import unittest
import pytest
import numpy as np


class TestPrecomputed(unittest.TestCase):
    """Tests classes in common_methods"""
    @pytest.fixture(autouse=True)
    def initdir(self, tmpdir):        
        print(tmpdir)
        tmpdir.chdir() # change to pytest-provided temporary directory


    def test_cross_validation(self):
        from common import model_maker as mm
        from common import stats_summary as ss


        process = 10 # int(sys.argv[1])     # batch process
        name = "titanic" # str(sys.argv[2])        # sample name
        path = "../data/" # str(sys.argv[3])        # path where code lives
        boot_kfold = "kfold" #str(sys.argv[4])  # use bootstrap or kfold
        ensem_single = "ensemble" # str(sys.argv[5])# use ensemble or standard classifiers


        model_auc = mm.model_loader_batch(process, ensemble_single=ensem_single)[1]
        model_auc_names = mm.model_loader_batch(process, ensemble_single=ensem_single)[0]
        n_cycles = 10
        k_folds  = 10
        n_reps   = 5
        roc_area = "deci"
        if model_auc[3] == "absvm":
            roc_area = "absvm"


        # ("trad-single-rbf",  custom_svm(), "rbf",         "single",  "trad", mut_rate, "auc", "roulette", 0.0)

        # model, is_precom, is_single, roc_area,
        auc, prc, f1, rec, acc, gmn, time, n_class, n_train = ss.cross_validation(sample_name=name,
                                                                                  model=model_auc[1],
                                                                                  is_precom = model_auc[2]=="precomputed",
                                                                                  is_single = model_auc[3],
                                                                                  roc_area=roc_area,
                                                                                  selection=model_auc[3+1], GA_mut=model_auc[4+1], GA_score=model_auc[5+1],
                                                                                  GA_selec=model_auc[6+1], GA_coef=model_auc[7+1], kfolds=k_folds, n_reps=n_reps, path=path)

        print(len(auc))
        print(len(prc))
        print(len(f1))
        print(len(rec))
        print(len(acc))
        print(len(gmn))
        print(len(time))
        print(len(n_class))
        print(len(n_train))
        


        print("parrito")


if __name__ == '__main__':
    unittest.main()

