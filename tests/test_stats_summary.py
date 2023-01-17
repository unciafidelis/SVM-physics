import os
import unittest
import pytest


class TestStatsSummary(unittest.TestCase):
    """
    Tests methods in stat_summary
    """
    @pytest.fixture(autouse=True)
    def initdir(self, tmpdir):        
        print(tmpdir)
        tmpdir.chdir() # change to pytest-provided temporary directory

    def test_cross_validation(self):
        import os
        import pandas as pd
        import datetime
        from common import model_maker as mm
        from common import stats_summary as ss

        process = 10 # batch process
        name = "titanic"
        path = "."
        boot_kfold = "kfold"
        exotic_single = "exotic"

        model_auc = mm.model_loader_batch(process, exotic_single=exotic_single)[1]
        model_auc_names = mm.model_loader_batch(process, exotic_single=exotic_single)[0]
        n_cycles = 2
        k_folds  = 3
        n_reps   = 2
        roc_area = "deci"
        if model_auc[3] == "absvm":
            roc_area = "absvm"

        start = datetime.datetime.now()
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
        dir_name_csv ="./results/stats_results_tests/"+name+"/"+boot_kfold+"/"
        if not os.path.exists(dir_name_csv):
            os.makedirs(dir_name_csv)            
        name_csv = dir_name_csv + model_auc[0]+"_1_"+boot_kfold+".csv" 
        df.to_csv(str(name_csv), index=False)
        
        end = datetime.datetime.now()
        elapsed_time = end - start
        print("Elapsed time = " + str(elapsed_time))
        print(model_auc[0], name)
        print(df)
        self.assertEqual(len(auc), int(k_folds * n_reps))
        self.assertEqual(len(prc), int(k_folds * n_reps))
        self.assertEqual(len(f1), int(k_folds * n_reps))
        self.assertEqual(len(rec), int(k_folds * n_reps))
        self.assertEqual(len(acc), int(k_folds * n_reps))
        self.assertEqual(len(gmn), int(k_folds * n_reps))
        self.assertEqual(len(time), int(k_folds * n_reps))
        self.assertEqual(len(n_class), int(k_folds * n_reps))
        self.assertEqual(len(n_train), int(k_folds * n_reps))


if __name__ == '__main__':
    unittest.main()
