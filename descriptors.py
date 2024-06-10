#%%

# import statements and global variables
import numpy as np
import pandas as pd
import os
import rdkit
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

TESTED_CSV = 'tested_molecules.csv'

#%%
# load dataset

tested_molecules = pd.read_csv(TESTED_CSV)

desc_list = [n[0] for n in Descriptors._descList]
calc = MoleculeDescriptors.MolecularDescriptorCalculator(desc_list)

tested_molecules['rdkit_mol'] = tested_molecules['SMILES'].apply(lambda x: rdkit.Chem.MolFromSmiles(x))
tested_molecules['ECFP'] = tested_molecules['rdkit_mol'].apply(lambda x: AllChem.GetMorganFingerprintAsBitVect(x, 3, nBits=2048))
desc = [calc.CalcDescriptors(x) for x in tested_molecules['rdkit_mol']]
tested_molecules['Descriptors'] = desc
print(tested_molecules.head())

#%%
for mol in tested_molecules['rdkit_mol'][1100:1115]:
    image = rdkit.Chem.Draw.MolToImage(mol)
    display(image)

