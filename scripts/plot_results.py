import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import os


sample_list = ['titanic', 'cancer', 'german', 'heart', 'solar', 'car', 'ecoli', 'wine', 'abalone'] # 'connect','adult']
color = ['red','blue','green','black','orange','brown','magenta','peru','indigo'] #,'cyan','slategray']
mark = ['P','s','^','>','h','o','v','+','<','p','x']
workpath = os.getcwd()
print(workpath)
path = workpath + "/results/stats_results/titanic/kfold"
dire_list = os.listdir(path)

p_file = []
file_list = []
for dire in dire_list:
    f = os.path.splitext(dire)[0]
    if f[0:3]=='gen' or f[0:4]=='trad':
        p_file.append(f.replace('_kfold', ''))
        file_list.append(dire)

p_file_data_frame = pd.DataFrame(p_file, columns = ['name'])


def plotmetric(metric_list, name, share):
    fig, (ax1, ax2, ax3) = plt.subplots(3,figsize=(14.5,8), sharex=True,sharey=share)
    q = 0
    axes = [ax1, ax2, ax3] # interates "q"
    for metric in metric_list:
        #plt.rcParams["figure.figsize"] = [15, 5]
        #plt.rcParams["figure.autolayout"] = True
        #fig=plt.figure()
        #fig=plt.figure(figsize=(15,5))
        #plt.subplot(idx[q])
        dt_mean = []
        k = 0
        for sample in sample_list:
            path = workpath + '/results/stats_results/' + sample + '/kfold'
            dt2 = pd.DataFrame()
            data_frame = pd.DataFrame()
            t = pd.DataFrame()
            dir_list = os.listdir(path)
            for filename in file_list:
                dft = pd.read_csv(workpath + '/results/stats_results/' + sample + '/kfold' + '/' + filename, index_col=None)
                col = dft.columns
                data_mean = pd.DataFrame(dft.mean())
                t = pd.concat([t, data_mean.loc[metric]])
            dt2 = pd.concat([dt2, p_file_data_frame])
            dt2.insert(0,'t', t.values)
            axes[q].plot(dt2["name"].to_numpy(), dt2['t'].to_numpy(), color[k], marker=mark[k], markersize=2, linewidth=1.0)
            k = k + 1
        if metric=='time':
            axes[q].set_yscale('log')
            axes[q].set_ylabel('Training Time [s]', fontsize=11)
        elif metric=='n_train':
            axes[q].set_yscale('log')
            axes[q].set_ylabel('Number of Training Vectors', fontsize=11)
        elif metric=='n_base':
            #axes[q].set_yscale('log')
            axes[q].set_yticks(np.arange(0.0, 8000, step=500))
            axes[q].set_ylabel('Number of Classifiers', fontsize=11)
        else:
            axes[q].set_yticks(np.arange(0.0, 1.2, step=0.1))
            axes[q].set_ylabel(metric.upper(), fontsize=12.5)
            axes[q].tick_params(direction='in')
            axes[q].tick_params(right=True, top=True)     
        q = q + 1
        location = 0
        plt.xticks(rotation=90)
        plt.xticks(fontsize=11.5)
        plt.yticks(fontsize=10.3)
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.1,right=0.85)  
        #fig.subplots_adjust(right=0.85)  
        plt.legend(sample_list, prop={"size":14},loc='lower right', bbox_to_anchor=(1.16, 0.7), frameon=False)
        #  fig.tight_layout()
        #  plt.savefig(u+'subplot.pdf')
        if not os.path.exists(workpath +"/plots/"):
            os.makedirs(workpath +"/plots/")
        plt.savefig(workpath +"/plots/" +name+'combined.pdf')


met = ['acc', 'prc', 'auc']
ric = ['time', 'n_base', 'n_train']
plotmetric(met, 'first', False)
plotmetric(ric, 'second', False)
