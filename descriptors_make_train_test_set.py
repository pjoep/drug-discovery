'''
This script is used to calculate 2D descriptors for the tested molecules, write them to a csv file
and then perform dimesionality reduction using KNIME. After that, the reduced dataset 
is loaded back to this script to create the X_train, X_test, y_train and y_test datasets.

You could also skip the KNIME dimensionality reduction part by loading
the KNIME_CSV file from the github repository.'''

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
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

TESTED_CSV = 'tested_molecules.csv'
KNIME_CSV = 'KNIME_filtered_descriptors.csv'
RANDOM_STATE = 42
TEST_SIZE = 0.2

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

# set up rdkit molecules and calculate fingerprints
tested_molecules['rdkit_mol'] = tested_molecules['SMILES'].apply(lambda x: rdkit.Chem.MolFromSmiles(x))
tested_molecules['ECFP4'] = tested_molecules['rdkit_mol'].apply(lambda x: list(AllChem.GetMorganFingerprintAsBitVect(x, 2, nBits=2048)))
tested_molecules['ECFP6'] = tested_molecules['rdkit_mol'].apply(lambda x: list(AllChem.GetMorganFingerprintAsBitVect(x, 3, nBits=2048)))
tested_molecules['MACCS'] = tested_molecules['rdkit_mol'].apply(lambda x: list(AllChem.GetMACCSKeysFingerprint(x)))

# calculate 2D descriptors
desc = [list(calc.CalcDescriptors(x)) for x in tested_molecules['rdkit_mol']]

# add 2D descriptors to dataframe
for descriptor in desc_list:
    tested_molecules[descriptor] = [x[desc_list.index(descriptor)] for x in desc]

#%%
''' write descriptors to csv file '''

tested_molecules.to_csv('tested_molecules_with_descriptors.csv', index=False)
#%%
''' load KNIME reduced dataset and create X_train, X_test, y_train and y_test datasets'''

# load KNIME reduced dataset
knime_filtered = pd.read_csv(KNIME_CSV)

# get all columns back to dataframe that where not passed by KNIME and 
# create new columns for filtered descriptors
knime_filtered['SMILES'] = tested_molecules['SMILES']
knime_filtered['PKM2_inhibition'] = tested_molecules['PKM2_inhibition']
knime_filtered['ERK2_inhibition'] = tested_molecules['ERK2_inhibition']
knime_filtered['ECFP4'] = tested_molecules['ECFP4']
knime_filtered['ECFP6'] = tested_molecules['ECFP6']
knime_filtered['MACCS'] = tested_molecules['MACCS']
knime_filtered['physiochemical_descriptors'] = knime_filtered.apply(
    lambda row: [row[desc] for desc in phc_desc_list if desc in knime_filtered.columns], 
    axis=1
)
knime_filtered['counter_descriptors'] = knime_filtered.apply(
    lambda row: [row[desc] for desc in count_desc_list if desc in knime_filtered.columns], 
    axis=1
)
knime_filtered['all_descriptors'] = knime_filtered['physiochemical_descriptors'] + knime_filtered['counter_descriptors']

# create X_train, X_test, y_train and y_test datasets

x_sets_list = ['physiochemical_descriptors', 'counter_descriptors', 'all_descriptors', 'ECFP4', 'ECFP6', 'MACCS']
y_sets_list = ['PKM2_inhibition', 'ERK2_inhibition']

for y_set in y_sets_list:
    for x_set in x_sets_list:
        x_vectors = np.array([np.array(x) for x in knime_filtered[x_set]])
        y_vectors = knime_filtered[y_set].values

        x_train, x_test, y_train, y_test = train_test_split(x_vectors, y_vectors, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True)




