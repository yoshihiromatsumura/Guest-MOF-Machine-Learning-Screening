import pandas as pd
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
import lazypredict
from lazypredict.Supervised import LazyRegressor
import matplotlib.pyplot as plt

def _return_aas(fdat, aas):
    f = open(fdat,'r')
    datalist = f.readlines()
    f.close()
    for num in range(len(datalist)):
        toks = datalist[num].split(' ')
        list_x = [a.rstrip().rstrip(':').rstrip(')').lstrip('(') for a in toks if a != '']
    for i,j in enumerate(list_x):
        aas[i]=j
    return aas

df = pd.read_excel('Dataset_I_35.xlsx', header=1, sheet_name=1, index_col=0)
dfx = pd.read_csv('2Dx_mord_A.csv')
lst_0 = ['AATS1s', 'AATSC2c', 'GATS3c', 'GATS3s', 'BCUTv-1l', 'SM1_Dzv', 'RNCG',
         'AETA_alpha', 'ETA_dAlpha_B', 'ETA_dPsi_A', 'nHBAcc', 'VSA_EState3']
X = dfx[lst_0].values
#scaler = StandardScaler().fit(X)
#X = scaler.transform(X)
y = df['Guest ratio  (from 1HNMR)'].values

list_yt = [] 
list_yp = []
list_ab = []
for i in range(41):
    list_yp.append([])
    list_ab.append([])
reg=LazyRegressor(verbose=0,ignore_warnings=False,predictions=True)
loo = LeaveOneOut()
for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    models, predictions=reg.fit(X_train,X_test,y_train,y_test)
    list_yt.append(*y_test)
    ttt = list(predictions.iloc[0,:].values)
    for i in range(len(ttt)):
        list_yp[i].append(ttt[i])
        list_ab[i].append(abs(y_test[0]-ttt[i]))
dfp = pd.DataFrame(predictions.columns)
list_m = []
list_r = []
list_x = []
list_w1 = []
list_w2 = []
for i in range(41):
    rmse_v = round(mean_squared_error(list_yt,list_yp[i],squared=False),3)
    r2s_v = round(r2_score(list_yt, list_yp[i]),3)
    mae_v = round(mean_absolute_error(list_yt, list_yp[i]),4)
    list_m.append(rmse_v)
    list_r.append(r2s_v)
    list_x.append(mae_v)
    list_w1.append(np.median(list_ab[i]))
    list_w2.append(np.std(list_ab[i]))
dfp['mae'] = list_x
dfp['std'] = list_w2
dfp['median'] = list_w1
dfp['rmse'] = list_m
dfp['r2s'] = list_r
list_0 = list(dfp[0].values)
dfp = dfp.sort_values('mae', ascending=True)
list_1 = list(dfp[0].values)
#dfp = dfp.reset_index(drop=True)
print(dfp)
print(sorted(list_x))
nnn=35; mmm=41
print('#nnn: ',nnn)
aas=np.zeros((nnn))
aas=_return_aas('abs_ob_mrd2Dx_A.dat', aas)
aabs=np.zeros((nnn,mmm))
for i in range(mmm):
    ii=list_1[i]
    ii=list_0.index(ii)
    if i==0:
        for j,k in enumerate(aas):
            aabs[j][i]=k
    else: 
        for j,k in enumerate(list_ab[ii]):
            aabs[j][i]=k

points = (aabs[:])
fig, ax = plt.subplots()
#bp = ax.boxplot(points,showmeans=True, meanline=True)
bp = ax.boxplot(points,showmeans=True,
                patch_artist=True,  #
                widths=0.6,  #
                boxprops=dict(facecolor='#1E90FF80',  #
                              color='black', linewidth=0.5),  #
                medianprops=dict(color='black', linewidth=0.5),  #
                meanprops=dict(marker = 'v', markersize= 1.5),
                whiskerprops=dict(color='black', linewidth=0.5),  #
                capprops=dict(color='black', linewidth=0.5),  #
                flierprops=dict(markeredgecolor='black', markeredgewidth=0.5,
                markersize=3.0)  #
                )
labels=list(list_1[0:mmm])
ax.set_xticklabels(labels)
plt.ylim(-0.01,1.25)
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
plt.savefig("mae_lzt_x.pdf")


