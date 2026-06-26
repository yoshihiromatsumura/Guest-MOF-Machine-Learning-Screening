import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
import umap
import time
start = time.time()

dfx = pd.read_csv('to_csv_out_dfxp.csv')
X = dfx.iloc[:,1:].values
scaler = StandardScaler().fit(X)
X = scaler.transform(X)
dfy = pd.read_csv('to_csv_out_y0P.csv')
y = dfy.iloc[:,1].values

## UMAP
umap = umap.UMAP(n_components=2, n_neighbors=10, random_state=73)
umap_x = umap.fit_transform(X)
##
plt.rcParams["font.size"] = 12
plt.scatter(umap_x[:,0], umap_x[:,1], c = y, cmap = "jet", 
            edgecolor = "None", marker='o', s=3, alpha=1.0)
plt.colorbar()

#ltmp = pd.read_csv('file_subst').values
#labels = [str(j[0]) for j in ltmp]
#for i in range(len(labels)):
#    print(labels[i],y[i],umap_x[i,0],umap_x[i,1])
#for i, label in enumerate(labels):
#    plt.text(umap_x[i,0],umap_x[i,1],label,size='x-small')

dfu = pd.DataFrame(umap_x)
print(dfu)
dfu.to_csv('for_tvmap_plus.csv')

plt.show()
plt.savefig("plt_umap_mrd.pdf")

print(time.time()-start)


