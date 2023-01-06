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
        import sys
        import os
        import pandas as pd
        import datetime
        from common import model_maker as mm
        from common import stats_summary as ss


        process = 10 # int(sys.argv[1])     # batch process
        name = "titanic" # str(sys.argv[2])        # sample name
        path = "../data/" # str(sys.argv[3])        # path where code lives
        boot_kfold = "kfold" #str(sys.argv[4])  # use bootstrap or kfold
        ensem_single = "ensemble" # str(sys.argv[5])# use ensemble or standard classifiers

        model_auc = mm.model_loader_batch(process, ensemble_single=ensem_single)[1]
        model_auc_names = mm.model_loader_batch(process, ensemble_single=ensem_single)[0]
        n_cycles = 2
        k_folds  = 2
        n_reps   = 2
        roc_area = "deci"
        if model_auc[3] == "absvm":
            roc_area = "absvm"


        # ("trad-single-rbf",  custom_svm(), "rbf",         "single",  "trad", mut_rate, "auc", "roulette", 0.0)
        start = datetime.datetime.now()
        # model, is_precom, is_single, roc_area,
        auc, prc, f1, rec, acc, gmn, time, n_class, n_train = ss.cross_validation(sample_name=name,
                                                                                  model=model_auc[1],
                                                                                  is_precom = model_auc[2],
                                                                                  kernel_fcn = model_auc[3],
                                                                                  roc_area=roc_area,
                                                                                  selection=model_auc[3+1], GA_mut=model_auc[4+1], GA_score=model_auc[5+1],
                                                                                  GA_selec=model_auc[6+1], GA_coef=model_auc[7+1], kfolds=k_folds, n_reps=n_reps, path=path)

        col_auc = pd.DataFrame(data=auc,    columns=["auc"])
        col_prc = pd.DataFrame(data=prc,    columns=["prc"])
        col_f1  = pd.DataFrame(data=f1,     columns=["f1"])
        col_rec = pd.DataFrame(data=rec,    columns=["rec"])
        col_acc = pd.DataFrame(data=acc,    columns=["acc"])
        col_gmn = pd.DataFrame(data=gmn,    columns=["gmn"])
        col_time= pd.DataFrame(data=time,   columns=["time"])
        col_base= pd.DataFrame(data=n_class,columns=["n_base"])
        col_size= pd.DataFrame(data=n_train,columns=["n_train"])
        df = pd.concat([col_auc["auc"], col_prc["prc"], col_f1["f1"], col_rec["rec"], col_acc["acc"], col_gmn["gmn"], col_time["time"], col_base["n_base"], col_size["n_train"]],
                       axis=1, keys=["auc", "prc", "f1", "rec", "acc", "gmn", "time", "n_base", "n_train"])

        dir_name_csv = path+"/stats_results_single/"+name+"/"+boot_kfold+"/"

        if not os.path.exists(dir_name_csv):
            os.mkdir(dir_name_csv)
            
        name_csv = dir_name_cs + model_auc[0]+"_1_"+boot_kfold+".csv" 
        df.to_csv(str(name_csv), index=False)
        

        end = datetime.datetime.now()
        elapsed_time = end - start
        print("Elapsed time = " + str(elapsed_time))
        print(model_auc[0], name)
        print(df)
        print('All names')
        for i in range(len(model_auc_names)):
            print(model_auc_names[i][0])


        print(len(auc))
        print(len(prc))
        print(len(f1))
        print(len(rec))
        print(len(acc))
        print(len(gmn))
        print(len(time))
        print(len(n_class))
        print(len(n_train))        


if __name__ == '__main__':
    unittest.main()

