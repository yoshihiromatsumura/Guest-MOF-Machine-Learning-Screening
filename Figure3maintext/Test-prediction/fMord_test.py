import pandas as pd
import pubchempy as pcp
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw, Descriptors, PandasTools, Descriptors3D
from mordred import Calculator, descriptors
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('to_csv_out_test.csv')
df['MOL'] = df['smiles'].apply(Chem.MolFromSmiles)
calc = Calculator(descriptors, ignore_3D=True)
df_mord = calc.pandas(df['MOL'])
df_mord2 = df_mord.select_dtypes("number")

X = df_mord2.iloc[:,:].values
scaler = StandardScaler().fit(X)
X = scaler.transform(X)
scaler_X = StandardScaler().fit(X)
clist = df_mord2.columns
for i in range(len(scaler_X.mean_)):
    if scaler_X.var_[i] < 1e-5:
        df_mord2 = df_mord2.drop(clist[i], axis=1)
print(df_mord2)
print(len(df_mord2.columns))
df_mord2.to_csv('t_mrd2D.csv')

