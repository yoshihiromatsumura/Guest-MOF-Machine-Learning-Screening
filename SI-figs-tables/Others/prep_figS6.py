import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
import umap
from matplotlib.ticker import MultipleLocator

nfs=6
df = pd.read_excel('Dataset_I_42.xlsx', header=1, sheet_name=1, index_col=0)
dfx = pd.read_csv('2Dx_mord_A.csv')
lst_0 = ['AATS1s', 'AATSC2c', 'GATS3c', 'GATS3s', 'BCUTv-1l', 'SM1_Dzv', 'RNCG',
         'AETA_alpha', 'ETA_dAlpha_B', 'ETA_dPsi_A', 'nHBAcc', 'VSA_EState3']
X = dfx[lst_0].values
scaler = StandardScaler().fit(X)
X = scaler.transform(X)
y = dfx[lst_0[nfs]].values
#y = y.reshape(-1,1)
#scaler_y = StandardScaler().fit(y)
#y = scaler_y.transform(y)
#y = y.reshape(1,-1)[0]

nnn=20
## UMAP
umap = umap.UMAP(n_components=2, n_neighbors=nnn, random_state=1)
umap_x = umap.fit_transform(X)
##
plt.scatter(umap_x[:,0], umap_x[:,1], c = y, cmap = "jet", 
            edgecolor = "None", marker='o', s=50, alpha=1.0)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=18)
plt.tick_params(labelsize=18)
plt.xlim(-8.2, -2.2)
plt.ylim(-1.2, 3.2)
plt.gca().yaxis.set_major_locator(MultipleLocator(1.0))
plt.gca().xaxis.set_major_formatter(plt.FormatStrFormatter('%.0f'))
plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.0f'))

ltmp = pd.read_csv('file_subst').values
labels = [str(j[0]) for j in ltmp]
with open('dat_'+str(nfs),'w') as f:
    for i in range(len(labels)):
        print(labels[i],y[i],umap_x[i,0],umap_x[i,1],file=f)
for i, label in enumerate(labels):
    plt.text(umap_x[i,0],umap_x[i,1],label,size='x-small')
plt.show()
plt.savefig("pfa_umfs_"+str(nfs)+".pdf")


