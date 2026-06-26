import pandas as pd
import pubchempy as pcp
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdMolDescriptors
from sklearn.preprocessing import StandardScaler

df = pd.read_excel('Dataset_I_35.xlsx', header=1, sheet_name=1, index_col=0)
dfc = df[['CAS No.']].values
properties = ['ConnectivitySMILES']
infos = []
for i in dfc:
    info = pcp.get_properties(properties, i, 'name')
    infos.append(info)
smileses = []
for i in np.arange(len(infos)):
    try:
        smiles = infos[i][0]["ConnectivitySMILES"]
        smileses.append(smiles)
    except:
        smiles = 'NaN'
        smileses.append(smiles)
df["SMILES"] = smileses
df['MOL'] = df['SMILES'].apply(Chem.MolFromSmiles)
fp = [AllChem.GetMACCSKeysFingerprint(mol) for mol in df['MOL'].values]
fps = []
for i in range(len(fp)):
    fps.append(list(np.array(fp[i], int)))
dfs = pd.DataFrame(data=fps)
print(dfs)

X = dfs.iloc[:,:].values
scaler = StandardScaler().fit(X)
X = scaler.transform(X)
scaler_X = StandardScaler().fit(X)
clist = dfs.columns
for i in range(len(scaler_X.mean_)):
    if scaler_X.var_[i] < 1e-5:
        dfs = dfs.drop(clist[i], axis=1)
print(dfs)
print(len(dfs.columns))
dfs.to_csv('tMaccs_0.csv')

