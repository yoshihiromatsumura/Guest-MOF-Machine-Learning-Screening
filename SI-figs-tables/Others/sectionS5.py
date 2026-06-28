import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from scipy import stats
from scipy.stats import rankdata
import time
start = time.time()

df = pd.read_excel('Dataset_I_42.xlsx', header=1, sheet_name=1, index_col=0)
dfx = pd.read_csv('2Dx_mord_A.csv')
lst_0 = ['AATS1s', 'AATSC2c', 'GATS3c', 'GATS3s', 'BCUTv-1l', 'SM1_Dzv', 'RNCG',
         'AETA_alpha', 'ETA_dAlpha_B', 'ETA_dPsi_A', 'nHBAcc', 'VSA_EState3']
X = dfx[lst_0].values
y = df['Guest ratio  (from 1HNMR)'].values

nnn=10000
print("#nnn: ",nnn)
forest = ExtraTreesRegressor(n_estimators=nnn,
                             criterion='squared_error', 
                             max_depth=None, 
                             min_samples_split=2, 
                             min_samples_leaf=1, 
                             min_weight_fraction_leaf=0.0, 
                             max_leaf_nodes=None, 
                             min_impurity_decrease=0.0, 
                             bootstrap=True, 
                             oob_score=False, 
                             n_jobs=None, 
                             random_state=1, 
                             verbose=0, 
                             warm_start=False, 
                             ccp_alpha=0.0, 
                             max_samples=None
                             )
forest.fit(X, y)
print('SCORE with selected Features: %1.2f' % forest.score(X, y))
print(forest.feature_importances_)
print("")

dfo = pd.read_excel('Dataset_I_9.xlsx', header=1, sheet_name=1, index_col=0)
#from Dataset_I_12.xlsx
dfv = pd.read_csv('vl_mrd2D.csv')
X_vl = dfv[lst_0].values
list_yt = dfo['Guest ratio  (from 1HNMR)'].values
list_yp = forest.predict(X_vl)

def catgB(list_x):
    tmp_x=[]
    for i,j in enumerate(list_x):
        if float(j)<1.0:
            tmp_x.append(0)
        else:
            tmp_x.append(1)
    return tmp_x
def catgC(list_x):
    tmp_x=[]
    for i,j in enumerate(list_x):
        if float(j)<0.5:
            tmp_x.append('a')
        elif float(j)<1.0:
            tmp_x.append('b')
        else:
            tmp_x.append('c')
    return tmp_x
def catgD1(list_x):
    tmp_x=[]
    for i,j in enumerate(list_x):
        if float(j)<0.25:
            tmp_x.append('a')
        elif float(j)<0.5:
            tmp_x.append('b')
        elif float(j)<1.0:
            tmp_x.append('c')
        else:
            tmp_x.append('d')
    return tmp_x
def catgD2(list_x):
    tmp_x=[]
    for i,j in enumerate(list_x):
        if float(j)<0.5:
            tmp_x.append('a')
        elif float(j)<1.0:
            tmp_x.append('b')
        elif float(j)<1.25:
            tmp_x.append('c')
        else:
            tmp_x.append('d')
    return tmp_x
def chkx(tmp_1,tmp_2):
    ii=0
    for i,j in enumerate(tmp_1):
        if j==tmp_2[i]:
            ii+=1
    return ii

print(list_yt)
print(list_yp)
print("")
print("#catgB: ")
tmp_ct = catgB(list_yt)
tmp_cp = catgB(list_yp)
print(tmp_ct)
print(tmp_cp)
print("chkx:",chkx(tmp_ct,tmp_cp),round(accuracy_score(tmp_ct,tmp_cp),3),
      "f1:",round(f1_score(tmp_ct,tmp_cp),3))
print("")
print("#catgC: ")
tmp_ct = catgC(list_yt)
tmp_cp = catgC(list_yp)
print(tmp_ct)
print(tmp_cp)
mmm = chkx(tmp_ct,tmp_cp)
print("chkx:",mmm,round(mmm/len(tmp_ct),3))
print("")
print("#catgD1: ")
tmp_ct = catgD1(list_yt)
tmp_cp = catgD1(list_yp)
print(tmp_ct)
print(tmp_cp)
mmm = chkx(tmp_ct,tmp_cp)
print("chkx:",mmm,round(mmm/len(tmp_ct),3))
print("")
print("#catgD2: ")
tmp_ct = catgD2(list_yt)
tmp_cp = catgD2(list_yp)
print(tmp_ct)
print(tmp_cp)
mmm = chkx(tmp_ct,tmp_cp)
print("chkx:",mmm,round(mmm/len(tmp_ct),3))
print("")

print("#rank: ")
print(list_yt)
print(list_yp)
tmp_rt = rankdata(list_yt)
tmp_rp = rankdata(list_yp)
print(*tmp_rt)
print(*tmp_rp)
res = stats.spearmanr(tmp_rt,tmp_rp)
print(round(res.statistic,3),f"{res.pvalue:.2e}")
res = stats.kendalltau(tmp_rt,tmp_rp)
print(round(res.statistic,3),f"{res.pvalue:.2e}")

print("")
print(time.time()-start)
print("")

