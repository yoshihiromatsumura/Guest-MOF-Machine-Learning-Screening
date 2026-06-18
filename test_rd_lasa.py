import pandas as pd
#from sklearn.ensemble import RandomForestRegressor
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import r2_score
#from sklearn.metrics import mean_squared_error
#import matplotlib.pyplot as plt
import pubchempy as pcp
import numpy as np
from IPython.display import SVG
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors, PandasTools
from rdkit.Chem.MolSurf import LabuteASA

df = pd.read_excel('230418_MOF_Guest_list.xlsx', header=1, sheet_name=1, index_col=0)
dfc = df[['CAS No.']].values
properties = ['CanonicalSMILES']
infos = []
for i in dfc:
    info = pcp.get_properties(properties, i, 'name')
    infos.append(info)
smileses = []
for i in np.arange(len(infos)):
    try:
        smiles = infos[i][0]["CanonicalSMILES"]
        smileses.append(smiles)
    except:
        smiles = "NaN"
        smileses.append(smiles)
df["SMILES"] = smileses
#PandasTools.AddMoleculeColumnToFrame(df, "SMILES")
#PandasTools.SaveXlsxFromFrame(df.fillna(0), "test_rd_0.xlsx", size=(150, 150))

infod = []
for i in df["SMILES"]:
    mol = Chem.AddHs(Chem.MolFromSmiles(i))
# 立体配座は不要だが、水素原子の有無で値が変わる。
    lasa = LabuteASA(mol)
    infod.append(lasa)
df["LASA"] = infod
PandasTools.AddMoleculeColumnToFrame(df, "SMILES")
PandasTools.SaveXlsxFromFrame(df.fillna(0), "test_rd_lasa.xlsx", size=(150, 150))

