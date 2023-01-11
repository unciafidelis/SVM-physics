import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import os


sample_list = ['titanic','cancer','german','heart','solar','car','ecoli','wine','abalone']
metric_list = ['auc','prc','time','acc','f1','gmn','rec']
color = ['red','blue','green','black','orange','brown','magenta','peru','indigo']
mark = ['P','s','^','>','h','o','v','+','<']

workpath = os.getcwd()
print(workpath)



for metric in metric_list:
  k = 0
  dt_mean = []
  plt.figure(figsize=(6,5))
  for sample in sample_list:
    file_list = []
    p_file = []
    path = workpath + '/' + sample + '/kfold'
    dt2 = pd.DataFrame()
    data_frame = pd.DataFrame()
    t = pd.DataFrame()
    dire_list = os.listdir(path)
    for dire in dire_list:
      f = os.path.splitext(dire)[0]
      if f[0:3]=='gen' or f[0:4]=='trad':
           p_file.append(f.replace('_kfold',''))
           file_list.append(dire)
    for filename in file_list:
        dft = pd.read_csv(workpath + '/' + sample + '/kfold' + '/' + filename, index_col=None)
        col = dft.columns
        data_mean = pd.DataFrame(dft.mean())
        t = t.append(data_mean.loc[metric])
    dt2 = dt2.append(p_file)
    dt2.insert(1, 't', t)
    print(i, dt2.head())
    plt.plot(dt2[0], dt2['t'], color[k], marker=mark[k], markersize=3, linewidth=1.0)
    k = k+1
  plt.xticks(rotation =90)
  plt.xticks(fontsize=7)
  if metric=='time':
    plt.yscale('log')
    plt.ylabel('TIME [s]', fontsize=12)
  else:
    #plt.yticks([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
    plt.ylabel(metric.upper(), fontsize=12)
  location = 0
  plt.tight_layout()
  plt.subplots_adjust(right=0.83)  
  plt.legend(['titanic','cancer','german','heart','solar','car','ecoli','wine','abalone'],prop={"size":8.5},loc='lower right',
  bbox_to_anchor=(1.23, 0.0),
  frameon=False)
  plt.savefig(metric + 'plot.pdf')
