import pandas as pd
import numpy as np
from rdkit import Chem
from mordred import Calculator, descriptors
from sklearn.preprocessing import StandardScaler
import time
start = time.time()

df = pd.read_csv('to_csv_out_fA_300.csv')
df = df.drop('Unnamed: 0', axis=1)
df['MOL'] = df['smiles'].apply(Chem.MolFromSmiles)

calc = Calculator(descriptors, ignore_3D=True)
df_mord = calc.pandas(df['MOL'])
print(df_mord)
lst_0 = ['AATS1s', 'AATSC2c', 'GATS3c', 'GATS3s', 'BCUTv-1l', 'SM1_Dzv', 'RNCG',
         'AETA_alpha', 'ETA_dAlpha_B', 'ETA_dPsi_A', 'nHBAcc', 'VSA_EState3']
print(lst_0)
print(df_mord[lst_0])
df_lst0 = df_mord[lst_0]
dfc = df_lst0.copy()
dfc['smiles'] = df['smiles']

ii = 0
for i in range(df_lst0.shape[0]):
    if 'missing' in str(df_lst0.iloc[i,:]):
        print(i, df['smiles'][i])
        dfc = dfc.drop(ii)
    ii = ii+1
dfc2 = dfc.reset_index(drop=True)
print(dfc2)

print("")
X = dfc2.iloc[:,:12].values
scaler = StandardScaler()
scaler.fit(X)
for i in range(len(scaler.mean_)):
    print(scaler.mean_[i],scaler.var_[i])
print("")

dfc2.to_csv('2D_mord_fA_300_xS.csv')

print(time.time()-start)


