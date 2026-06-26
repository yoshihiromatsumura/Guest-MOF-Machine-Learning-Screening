import pandas as pd
import numpy as np
import time
start = time.time()

dfx = pd.read_csv('tAP_0.csv')#Anyothers#
dfx.index = np.arange(1, len(dfx)+1)
X = dfx.iloc[:,1:]
print(X)

res = X.corr()
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
print("")
print(res)
print("")

#tcrr= 1.0-1e-5
tcrr= 1.0-1e-15
nnn = len(res.columns)
list_s = []
list_ss = []
for i in range(nnn):
    if res.columns[i] not in list_s:
        list_s.append(res.columns[i])
        list_ss.append([])
        list_ss[-1].append(res.columns[i])
        for j in range(i+1,nnn):
            if abs(res.values[i][j]) > tcrr:
                list_s.append(res.columns[j])
                list_ss[-1].append(res.columns[j])
list_p = []
for i in range(len(list_ss)):
    if len(list_ss[i]) > 1:
        print(i,len(list_ss[i]),list_ss[i])
    list_p.append(list_ss[i][0])
print("")
print('num:',len(list_p))
print("")
df_mord2 = X[list_p]
print(df_mord2)
df_mord2.to_csv('tfeats_pc.csv')
print("")
print(time.time()-start)

