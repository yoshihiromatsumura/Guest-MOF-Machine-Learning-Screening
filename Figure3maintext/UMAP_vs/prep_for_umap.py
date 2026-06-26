import pandas as pd
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import numpy as np
from rdkit import Chem
from sklearn.ensemble import ExtraTreesRegressor
import time
start = time.time()

df = pd.read_excel('Dataset_I_42.xlsx', header=1, sheet_name=1, index_col=0)
dfx = pd.read_csv('2Dx_mord_A.csv')
lst_0 = ['AATS1s', 'AATSC2c', 'GATS3c', 'GATS3s', 'BCUTv-1l', 'SM1_Dzv', 'RNCG',
         'AETA_alpha', 'ETA_dAlpha_B', 'ETA_dPsi_A', 'nHBAcc', 'VSA_EState3']
X = dfx[lst_0].values
y = df['Guest ratio  (from 1HNMR)'].values
model = ExtraTreesRegressor(n_estimators=10000,
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
model.fit(X, y)
y_pred = model.predict(X)
print("")
print(mean_squared_error(y, y_pred, squared=False))
print(r2_score(y, y_pred))
print("")

dfp = pd.read_csv('2D_mord_fA_300_xS.csv')
dfp = dfp.drop('Unnamed: 0', axis=1)
XP = dfp[lst_0].values
y_P = model.predict(XP)
dfp["pred"] = y_P

dfxp = pd.concat([dfx[lst_0],dfp[lst_0]])
dfxp = dfxp.reset_index(drop=True)
print(dfxp)
dfxp.to_csv('to_csv_out_dfxp.csv')

y0P = pd.concat([pd.DataFrame(y_pred),pd.DataFrame(y_P)])
y0P = y0P.reset_index(drop=True)
print(y0P)
y0P.to_csv('to_csv_out_y0P.csv')

print(time.time()-start)
print("")


