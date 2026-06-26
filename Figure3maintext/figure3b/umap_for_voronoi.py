import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
import umap
import time
start = time.time()

df = pd.read_excel('Dataset_I_42.xlsx', header=1, sheet_name=1, index_col=0)
dfx = pd.read_csv('2Dx_mord_A.csv')
lst_0 = ['AATS1s', 'AATSC2c', 'GATS3c', 'GATS3s', 'BCUTv-1l', 'SM1_Dzv', 'RNCG',
         'AETA_alpha', 'ETA_dAlpha_B', 'ETA_dPsi_A', 'nHBAcc', 'VSA_EState3']
X = dfx[lst_0].values
scaler = StandardScaler().fit(X)
X = scaler.transform(X)
y = df['Guest ratio  (from 1HNMR)'].values

nnn=20
## UMAP
umap = umap.UMAP(n_components=2, n_neighbors=nnn, random_state=1)
umap_x = umap.fit_transform(X)
##
plt.rcParams["font.size"] = 12
plt.scatter(umap_x[:,0], umap_x[:,1], c = y, cmap = "jet", 
            edgecolor = "None", marker='o', s=50, alpha=1.0)
plt.colorbar()
#plt.xlim(0.5,5.5)
#plt.ylim(1.0,6.5)

ltmp = pd.read_csv('file_subst').values
labels = [str(j[0]) for j in ltmp]
for i in range(len(labels)):
    print(labels[i],y[i],umap_x[i,0],umap_x[i,1])
for i, label in enumerate(labels):
    plt.text(umap_x[i,0],umap_x[i,1],label,size='x-small')

plt.show()
plt.savefig("plt_umap_fs_"+str(nnn)+".pdf")

print(time.time()-start)


