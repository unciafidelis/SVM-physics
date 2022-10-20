import numpy as np
import pandas as pd
import multiprocessing
import copy
from sklearn.metrics import auc,roc_auc_score
from sklearn.inspection import permutation_importance
import shap
import pickle
import os


class ModelStoring:
    """
    Class that stores a trained model in a pickle file
    """
    def __init__(self, file_name="saved_model.pkl"):
        self.file_name = file_name

    def save_model(self, model=None):
        """
        Saves an input trained model to a pickle file
        """
        print("Saving the model into a pickle(binary) file")
        fo = open(self.file_name,'wb')
        pickle.dump(model, fo)
        fo.close()
        
    def load_model(self):
        """
        Returns the trained model from a pickle file
        """
        file_name = self.file_name
        with open(file_name, 'rb') as f:
            print("Loading model from a pickle(binary) file")
            return pickle.load(f)

    def delete_model(self):
        """
        Deletes the trained model stored in a pickle file
        """
        if os.path.exists(self.file_name):
            os.remove(self.file_name)
        else:
            print("The file does not exist")

def roc_curve_adaboost(Y_thresholds, Y_test) -> tuple:
    """
    Tailored function to create the TPR and FPR, for ROC curve
    """    
    if type(Y_test) != type(np.array([])): # check data format
        Y_test = Y_test.values
    TPR_list, FPR_list = [], []
    for i in range(Y_thresholds.shape[0]):
        tp,fn,tn,fp=0,0,0,0
        for j in range(Y_thresholds.shape[1]):             
            if(Y_test[j] == 1  and Y_thresholds[i][j] ==  1):  tp+=1
            if(Y_test[j] == 1  and Y_thresholds[i][j] == -1):  fn+=1
            if(Y_test[j] == -1 and Y_thresholds[i][j] == -1):  tn+=1
            if(Y_test[j] == -1 and Y_thresholds[i][j] ==  1):  fp+=1
        TPR_list.append( tp/(tp+fn) )
        FPR_list.append( fp/(tn+fp) )
        # sort the first list and map ordered indexes to the second list
    FPR_list, TPR_list = zip(*sorted(zip(FPR_list, TPR_list)))
    TPR = np.array(TPR_list)
    FPR = np.array(FPR_list)
    return (TPR, FPR)


class VariableImportance:
    """
    Class that asses the importance of a given variable when training a model
    Input data must be pandas data frames, main methods return an orderdered list of variables
    """
    def __init__(self, model, X_train, Y_train, X_test, Y_test, global_auc=1.0, roc_area="prob", workpath="."):
        self.m_model = model
        self.m_x_train = X_train
        self.m_y_train = Y_train
        self.m_x_test = X_test
        self.m_y_test = Y_test
        self.m_global_auc = global_auc
        self.m_roc_area = roc_area
        self.m_workpath = workpath

    def roc_for_variable_set(self, variable):
        """
        Returns the AUC for the model training/testing skipping "variable"
        """
        method  = self.m_model
        X_train = self.m_x_train.drop(variable, axis=1)
        Y_train = self.m_y_train
        X_test  = self.m_x_test.drop(variable, axis=1)
        Y_test  = self.m_y_test
        method.fit(X_train, Y_train)
        if self.m_roc_area=="absv":
            y_thresholds = method.decision_thresholds(X_test, glob_dec=True)
            TPR, FPR = roc_curve_adaboost(y_thresholds, Y_test)
            method.clean()
            return auc(FPR,TPR)
        elif self.m_roc_area=="prob":
            Y_pred_prob = method.predict_proba(X_test)[:,1]
            return roc_auc_score(Y_test, Y_pred_prob)
        elif self.m_roc_area=="deci":
            Y_pred_dec = method.decision_function(X_test)
            return roc_auc_score(Y_test, Y_pred_dec)

    def one_variable_out(self):
        """
        Calculate the importance using the loss in AUC if a variable is removed
        Returns the ordered list of variables with their corresponding AUC
        The first element in the returned list is the one that affected the least the model performance
        """
        p = multiprocessing.Pool(None, maxtasksperchild=1) # multi-thread processing
        variables = list(self.m_x_train.columns)
        results = p.map(self.roc_for_variable_set, variables) # [[v for v in method.variables if v != variable] for variable in method.variables])
        sorted_variables_with_results = list(sorted(zip(variables, results), key=lambda x: x[1]))
        p.close()
        return sorted_variables_with_results
        
    def recursive_one_variable_out(self):
        """
        Calculate the importance using the loss in AUC if a variable is removed recursively
        Returns the ordered list of variables with their corresponding AUC
        The first element in the returned list is the one that affected the least the model performance
        """
        p = multiprocessing.Pool(None, maxtasksperchild=1) # multi-thread processing
        variables = list(self.m_x_train.columns)
        results = p.map(self.roc_for_variable_set, variables)
        sorted_variables_with_results = list(sorted(zip(variables, results), key=lambda x: x[1])) 
        removed_variables_with_results = sorted_variables_with_results[:1]
        remaining_variables = [v for v, r in sorted_variables_with_results[1:]]
        while len(remaining_variables) > 1:
            results = p.map(self.roc_for_variable_set,
                            [[v for v in remaining_variables if v != variable] for variable in remaining_variables])
            sorted_variables_with_results = list(sorted(zip(remaining_variables, results), key=lambda x: x[1]))
            removed_variables_with_results += sorted_variables_with_results[:1]
            remaining_variables = [v for v, r in sorted_variables_with_results[1:]]
        removed_variables_with_results += sorted_variables_with_results[1:]
        p.close()
        return removed_variables_with_results
        
    def permutation_feature(self):
        """
        Calculate the importance using the permutation
        Returns the ordered list of variables with their corresponding AUC
        """
        results = permutation_importance(self.m_model, self.m_x_train, self.m_y_train, n_repeats=50, random_state=None)
        variables = list(self.m_x_train.columns)
        sorted_variables_with_results = list(sorted(zip(variables, results.importances_mean), key=lambda x: x[1], reverse=True))
        return sorted_variables_with_results

    def shapley_values(self):
        """
        Calculate variable importance using SHAPLEY
        """
        method  = self.m_model
        X_train = self.m_x_train
        Y_train = self.m_y_train
        X_test  = self.m_x_test
        Y_test  = self.m_y_test
        method.fit(X_train, Y_train)
        if self.m_roc_area=="absv":
            f = lambda x: method.decision_thresholds(x, glob_dec=True)
            method.clean()            
        elif self.m_roc_area=="prob":
            f = lambda x: method.predict_proba(x)[:,1]
        elif self.m_roc_area=="deci":
            f = lambda x: method.decision_function(x)            
        med = X_train.median().values.reshape((1,X_train.shape[1]))
        explainer = shap.Explainer(f, med)
        shap_values = explainer(X_test.iloc[0:1000,:])
        print(type(shap_values), len(shap_values))
        print(shap_values)
        import matplotlib.pyplot as plt
        fig = shap.plots.heatmap(shap_values, show=False) # shap.summary_plot(shap_values, X_test.iloc[0:1000,:])
        f = plt.gcf()
        f.savefig(self.m_workpath+"/files/heatmap.png")
        plt.close(f)


    def shapley_values_second(self):
        """
        Calculate second method the importance using the permutation SHAPLEY
        not well understood yet
        """        
        method = self.m_model
        X_train = self.m_x_train
        Y_train = self.m_y_train
        X_test  = self.m_x_test
        Y_test  = self.m_y_test

        method.fit(X_train, Y_train)
        if self.m_roc_area=="absv":
            f = lambda x: method.decision_thresholds(x, glob_dec=True)
            method.clean()            
        elif self.m_roc_area=="prob":
            f = lambda x: method.predict_proba(x)[:,1]
        elif self.m_roc_area=="deci":
            f = lambda x: method.decision_function(x)

        # second shapley values
        explainer = shap.KernelExplainer(method.decision_function, X_train)
        shap_values = explainer.shap_values(X_test.iloc[0:1000,:])
        print(type(shap_values), len(shap_values))
        print(shap_values)
        # shap.force_plot(explainer.expected_value[0], shap_values[0], X_test)

        # fig = shap.plots.heatmap(shap_values, show=False) # shap.summary_plot(shap_values, X_test.iloc[0:1000,:])
        # f = plt.gcf()
        # f.savefig('heatmap_dos.png')
        # plt.close(f)
        # shap.force_plot(explainer.expected_value[0], shap_values[0], X_test)
        print("hello world!")


class GridSearch:
    """
    Class that searches optimal model parameters
    """
    def __init__(self, model):
        self.m_model = model

    def cross_validation(self):
        # k-fold cross validation method
        kf = KFold(n_splits=5, shuffle=False, random_state=None)
        acc_arr = ([])
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]                
            model.fit(X_train, Y_train)
            acc_arr = np.append(acc_arr, self.m_model.score(X_test, Y_test))

        return acc_arr

        
def fom_simple(sig, bkg):
    a = np.add(sig, bkg)
    ss = 1/(np.sqrt(bkg))
    b = np.add(bkg,ss*ss)
    mul = np.multiply(bkg,bkg)
    sum = (a*b)/(np.add(mul,a*ss*ss))
    first = a*np.log(sum)
    c = (sig*ss*ss)/(bkg*b)
    c1 = np.add(1,c)
    clog = np.log(c1)
    sec = mul/(ss*ss)
    second = sec*clog
    final = 2*(first-second)
    fpt = np.sqrt(final)
    countf, binf = np.histogram(fpt,bins=100,range=[-2,2])
    #plt.plot(binf,fpt,marker='o',color='red')
    #plt.savefig('fom_new.png')
    return np.nanmax(fpt)


def fom_simple_max(sig, bkg):
    k= sig/np.sqrt(np.add(sig, bkg))
    countf, binf = np.histogram(k,bins=100,range=[-2,2])
    plt.plot(binf,k,marker='o',color='red')
    plt.savefig('fom_old.png')
    return np.nanmax(k)


def plot_roc_curve(TPR, FPR, sample="titanic", real="sorted", glob_local="global", name="name", kernel="rbf", nClass=10):
    """
    Method that plots AUC, saves images and table in CSV
    """
    import matplotlib.pyplot as plt
    if(real=='sorted'):
        TPR = np.sort(TPR,axis=None)
        FPR = np.sort(FPR,axis=None)
    if glob_local: glob_local='global'
    else:          glob_local='local'
    area = auc(FPR,TPR)
    plt.figure()
    lw = 2
    plt.plot(FPR, TPR, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f, N = %0.0f)'  %(area, nClass), linestyle="-", marker="o")
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve -' + sample)
    plt.legend(loc="lower right")
    if not os.path.exists('./plots/'):
        os.mkdir('./plots/')
    plt.savefig('./plots/roc_curve_'+sample+'_'+real+'_'+glob_local+'_'+name+'_'+kernel+'.png')
    output = pd.DataFrame({'False positive rate': FPR,'True positive rate': TPR, 'Area': area})
    if not os.path.exists('./output/' + sample + '/'):
        os.makedirs('./output/' + sample + '/')        
    output.to_csv('./output/' + sample +  '/' + 'BoostSVM_ROC.csv', index=False)
    plt.close()
