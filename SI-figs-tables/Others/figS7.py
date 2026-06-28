import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
import umap
from matplotlib.ticker import MultipleLocator

nfs = 11
vmin = -0.0440740740740741
vmax = 10.00648148148148
#8: 0.0 0.0642857142857143
#6: 0.0571324684540014 0.6938979238553065
#11: -0.0440740740740741 10.00648148148148
dfx = pd.read_csv('to_csv_out_dfxp.csv')
X = dfx.iloc[:,1:].values
scaler = StandardScaler().fit(X)
X = scaler.transform(X)
y = dfx.iloc[:,nfs+1].values
## UMAP
umap = umap.UMAP(n_components=2, n_neighbors=10, random_state=73)
umap_x = umap.fit_transform(X)
##
plt.rcParams["font.size"] = 16
plt.scatter(umap_x[:,0], umap_x[:,1], c = y, cmap = "jet", 
            edgecolor = "None", marker='o', s=3, alpha=1.0,
            vmin=vmin,vmax=vmax)
plt.colorbar()
plt.gca().xaxis.set_major_locator(MultipleLocator(5.0))
plt.show()
plt.savefig("plt_umfc_"+str(nfs)+".pdf")

