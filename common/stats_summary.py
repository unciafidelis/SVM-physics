"""
-----------------------------------------
 Authors: A. Ramirez-Morales
-----------------------------------------
"""
import numpy as np
import time

from sklearn.metrics import accuracy_score,auc,precision_score,roc_auc_score,f1_score,recall_score
from sklearn.model_selection import RepeatedKFold

# framework includes
from data.data_preparation import DataPreparation
from common.common_methods import roc_curve_adaboost
# import model_maker as mm
# import data_visualization as dv

from common.genetic_selection import GeneticSelection



def cross_validation(sample_name, model, is_precom, is_single, roc_area, selection, GA_mut=0.25, GA_score='', GA_selec='', GA_coef=0.5, kfolds=1, n_reps=1, path='.'):

    print(is_precom)
    print(is_single)
    input()
    
    # fetch data_frame without preparation
    data = DataPreparation(path)
    sample_df_temp = data.fetch_data(sample_name)
    train_test = type(sample_df_temp) is tuple  # are the data already splitted?
    if not train_test:
        sample_df = sample_df_temp
    else:
        sample_train_df, sample_test_df = sample_df_temp

    area_scores,prec_scores,f1_scores,recall_scores,acc_scores,gmean_scores,time_scores = ([]),([]),([]),([]),([]),([]),([])
    n_class_scores, n_train_scores = ([]), ([])
        
    X, Y = data.dataset(sample_name=sample_name, data_set=sample_df,
                        sampling=True, split_sample=0.0)
    
    # n-k fold cross validation, n_cycles = n_splits * n_repeats
    rkf = RepeatedKFold(n_splits = kfolds, n_repeats = n_reps, random_state = 1) # set random state=1 for reproducibility
    for i, (train_index, test_index) in enumerate(rkf.split(X)):
        X_train, X_test = X.loc[train_index], X.loc[test_index]
        Y_train, Y_test = Y.loc[train_index], Y.loc[test_index]
        start = time.time()
        # keep the chromosome size within the range [100,1000]
        sample_chromn_len = int(len(Y_train)*0.25)
        if sample_chromn_len > 1000:
            sample_chromn_len = 1000
        elif sample_chromn_len < 100:
            sample_chromn_len = 100
            
        if selection == 'gene': # genetic selection
            GA_selection = GeneticSelection(model, roc_area, X_train, Y_train, X_test, Y_test,
                                            pop_size=10, chrom_len=sample_chromn_len, n_gen=50, coef=GA_coef,
                                            mut_rate=GA_mut, score_type=GA_score, selec_type=GA_selec)
            GA_selection.execute()
            GA_train_indexes = GA_selection.best_population()
            X_train, Y_train, X_test, Y_test = data.dataset(sample_name=sample_name, indexes=GA_train_indexes)
            print(len(X_train), len(Y_test), len(GA_train_indexes), 'important check for GA outcome')
            print(len(Y_train[Y_train==1]), 'important check for GA outcome')

        if is_precom: # pre-compute the kernel matrices if requested
            matrix_train = matrix(X_train)
            matrix_test  = matrix(X_test)
            model.fit(matrix_train, Y_train)
        else:
            model.fit(X_train, Y_train)


        n_base_class = 0
        no_zero_classifiers = True
        if roc_area=="absv":
            n_base_class = model.n_classifiers
            if n_base_class==0:
                no_zero_classifiers = False
                
        if no_zero_classifiers:
            y_pred = model.predict(X_test)
            prec = precision_score(Y_test, y_pred)
            f1 = f1_score(Y_test, y_pred)
            recall = recall_score(Y_test, y_pred)
            acc = accuracy_score(Y_test, y_pred)
            gmean = np.sqrt(prec * recall)
            # calculate roc-auc depending on the classifier
            if roc_area=="absv":
                y_thresholds = model.decision_thresholds(X_test, glob_dec=True)
                TPR, FPR = roc_curve_adaboost(y_thresholds, Y_test)
                area = auc(FPR,TPR)
                model.clean()
            elif roc_area=="prob":
                Y_pred_prob = model.predict_proba(X_test)[:,1]
                area = roc_auc_score(Y_test, Y_pred_prob)
            elif roc_area=="deci":
                Y_pred_dec = model.decision_function(X_test)
                area = roc_auc_score(Y_test, Y_pred_dec)

            end = time.time()
            time_scores    = np.append(time_scores, end-start)
            area_scores    = np.append(area_scores, area)
            prec_scores    = np.append(prec_scores, prec)
            f1_scores      = np.append(f1_scores,   f1)
            recall_scores  = np.append(recall_scores, recall)
            acc_scores     = np.append(acc_scores, acc)
            gmean_scores   = np.append(gmean_scores, gmean)
            n_class_scores = np.append(n_class_scores, n_base_class)
            n_train_scores = np.append(n_train_scores, len(X_train))
        else: # this needs to be re-checked carefully
            end = time.time()
            time_scores    = np.append(time_scores, end-start)
            area_scores    = np.append(area_scores, 0)
            prec_scores    = np.append(prec_scores, 0)
            f1_scores      = np.append(f1_scores,   0)
            recall_scores  = np.append(recall_scores, 0)
            acc_scores     = np.append(acc_scores, 0)
            gmean_scores   = np.append(gmean_scores, 0)
            n_class_scores = np.append(n_class_scores, 0)
            n_train_scores = np.append(n_train_scores, len(X_train))

    return area_scores,prec_scores,f1_scores,recall_scores,acc_scores,gmean_scores,time_scores,n_class_scores,n_train_scores
