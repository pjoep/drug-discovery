#%%

''' import statements and global variables'''
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
from sklearn.preprocessing import StandardScaler

TESTED_CSV = 'tested_molecules.csv'
KNIME_CSV = 'KNIME_filtered_descriptors.csv'

#%%
''' load dataset to dataframe and calculate descriptors '''

tested_molecules = pd.read_csv(TESTED_CSV)

# list of all 2D descriptors
desc_list = [n[0] for n in Descriptors._descList]
# list of all physiochemical 2D descriptors 
phc_desc_list = [i for i in desc_list if not i.startswith('fr_')]
# list of all counter 2D descriptors
count_desc_list = [i for i in desc_list if i.startswith('fr_')]

# calculate 2D descriptor objects
calc = MoleculeDescriptors.MolecularDescriptorCalculator(desc_list)
phc_calc = MoleculeDescriptors.MolecularDescriptorCalculator(phc_desc_list)
count_calc = MoleculeDescriptors.MolecularDescriptorCalculator(count_desc_list)

# set up rdkit molecules and calculate fingerprints
tested_molecules['rdkit_mol'] = tested_molecules['SMILES'].apply(lambda x: rdkit.Chem.MolFromSmiles(x))
tested_molecules['ECFP'] = tested_molecules['rdkit_mol'].apply(lambda x: AllChem.GetMorganFingerprintAsBitVect(x, 3, nBits=2048))

# calculate 2D descriptors
desc = [list(calc.CalcDescriptors(x)) for x in tested_molecules['rdkit_mol']]
phc_desc = [phc_calc.CalcDescriptors(x) for x in tested_molecules['rdkit_mol']]
count_desc = [count_calc.CalcDescriptors(x) for x in tested_molecules['rdkit_mol']]

# add 2D descriptors to dataframe
tested_molecules['Descriptors'] = desc
tested_molecules['Phc_Descriptors'] = phc_desc
tested_molecules['Count_Descriptors'] = count_desc

for descriptor in desc_list:
    tested_molecules[descriptor] = [x[desc_list.index(descriptor)] for x in desc]


#%%

tested_molecules.to_csv('tested_molecules_with_descriptors.csv', index=False)



#%%

filter_descriptors = pd.read_csv(KNIME_CSV)
''' PCA analysis of 2D descriptors '''

# scale the descriptors
scaler = StandardScaler()
scaled_filter_descriptors = pd.DataFrame(scaler.fit_transform(filter_descriptors.iloc[:, 1:]), columns=filter_descriptors.columns[1:])
pca = PCA()
pca.fit(scaled_filter_descriptors.iloc[:, 1:])
pca_transform = pca.transform(scaled_filter_descriptors.iloc[:, 1:])
pca_df = pd.DataFrame(pca_transform, columns=['PC'+str(i) for i in range(1, pca_transform.shape[1]+1)])
pca_df['PKM2_inhibition'] = tested_molecules['PKM2_inhibition']
pca_df['ERK2_inhibition'] = tested_molecules['ERK2_inhibition']

fig, ax = plt.subplots(4,1, figsize=(10, 10))
sns.scatterplot(ax=ax[0], x=pca_df['PC1'], y=pca_df['PC2'], hue=pca_df['ERK2_inhibition'])
sns.barplot(ax=ax[1], x=np.arange(1, len(pca.explained_variance_ratio_)+1), y=pca.explained_variance_ratio_)
sns.barplot(ax=ax[2], x=np.arange(1, len(pca.explained_variance_ratio_)+1), y=np.cumsum(pca.explained_variance_ratio_))
sns.barplot(ax=ax[3], y=abs(pca.components_[0]), x=pca_df.columns[:-2])
plt.show()


