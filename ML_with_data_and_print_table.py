#%%

''' import statements and global variables'''
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import rdkit
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import recall_score
from sklearn.model_selection import cross_val_predict, train_test_split, cross_validate, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error
import copy

TESTED_CSV = r'tested_molecules.csv'
RANDOM_STATE = 42
TEST_SIZE = 0.01
DUPLICATION_TIMES = 1 #options are 1,2,3,4 5, 10 or None
#KNIME_CSV = f'KNIME_filtered_descriptors_{DUPLICATION_TIMES}x_duplicate.csv' #change to right duplication time!!
KNIME_CSV = 'KNIME_filtered_descriptors.csv' #use when no duplication is needed
#%%
''' load dataset to dataframe and calculate descriptors '''

tested_molecules = pd.read_csv(TESTED_CSV)

rows_to_duplicate_PKM = tested_molecules[tested_molecules['PKM2_inhibition'].eq(1)]
rows_to_duplicate_ERK = tested_molecules[tested_molecules['ERK2_inhibition'].eq(1)]
rows_to_duplicate = pd.concat([rows_to_duplicate_PKM, rows_to_duplicate_ERK], ignore_index=True)

#duplicate_times = DUPLICATION_TIMES
#duplicate_rows = pd.concat([rows_to_duplicate]*duplicate_times, ignore_index=True)

#tested_molecules = pd.concat([tested_molecules, duplicate_rows], ignore_index=True)



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

# perform scaling on the filtered descriptors

scaler = MinMaxScaler() # can be changed to StandardScaler but MinMaxScaler is used for better comparability

for desc in desc_list:
    if desc in knime_filtered.columns:
        knime_filtered[desc] = scaler.fit_transform(knime_filtered[desc].values.reshape(-1, 1))

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



#%%
# create X_train, X_test, y_train and y_test datasets
x_sets_list = ['physiochemical_descriptors', 'counter_descriptors', 'all_descriptors', 'ECFP4', 'ECFP6', 'MACCS']
y_sets_list = ['PKM2_inhibition', 'ERK2_inhibition']

# Default models
models = {"nnet": MLPClassifier(random_state=RANDOM_STATE, max_iter=100000, early_stopping=True, n_iter_no_change=100),
          "rf_class": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, oob_score=recall_score, n_jobs=-1),
          "xgb_clf": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_STATE, n_jobs=-1),
}
        
KFold = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

knime_filtered_seperate_pos = pd.DataFrame()
smiles_list = []
for i in knime_filtered.index:
    if i in range(1093, 1117):
        smiles_list.append(knime_filtered['SMILES'][i])
    if knime_filtered['SMILES'][i] in smiles_list:  
        knime_filtered_seperate_single = knime_filtered.loc[[i]]
        knime_filtered_seperate_pos = pd.concat([knime_filtered_seperate_pos, knime_filtered_seperate_single])

for index in knime_filtered_seperate_pos.index:
    print(index)
#    knime_filtered.drop(index, inplace=True)

smiles_list = []
for index in knime_filtered_seperate_pos.index:
    smiles_list.append(knime_filtered_seperate_pos['SMILES'][index])
print(len(smiles_list))

duplicates = []
for index in knime_filtered.index:
    if knime_filtered['SMILES'][index] in smiles_list:
        duplicates.append(index)

print(duplicates)
print(len(duplicates))

#%%
best_model_ERK=0
best_model_PKM=0
best_sensitivity_ERK=0
best_sensitivity_PKM=0
scores = {}
counter=0
for y_set in y_sets_list:
    scores[y_set] = {}
    for x_set in x_sets_list:
        scores[y_set][x_set] = {}

        x_vectors = np.array([np.array(x) for x in knime_filtered[x_set]])
        y_vectors = knime_filtered[y_set].values

        x_test_test = np.array([np.array(x) for x in knime_filtered_seperate_pos[x_set]])
        y_test_test = knime_filtered_seperate_pos[y_set].values

        x_train, x_test, y_train, y_test = train_test_split(x_vectors, y_vectors, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True)

        for m in models:
            counter= counter+1

            #cross-valid
            #scores, trained_model = cross_validate(models[m], x_vectors, y_vectors, cv=KFold, n_jobs=-1, return_estimator=True)
            #sensitivity = recall_score(y_vectors, predictions)
            #specificity = recall_score(y_vectors, predictions, pos_label=0)
            #predictions_test_test = models[m].predict(x_test_test)
            #sensitivity_test_test = recall_score(y_test_test, predictions_test_test)
            #specificity_test_test = recall_score(y_test_test, predictions_test_test, pos_label=0)
            #scores[y_set][x_set][m + "_sensitivity"] = sensitivity
            #scores[y_set][x_set][m + "_specificity"] = specificity
            #scores[y_set][x_set][m + "_sensitivity_test_test"] = sensitivity_test_test
            #cores[y_set][x_set][m + "_specificity_test_test"] = specificity_test_test
            #scores[y_set][x_set][m + "_mse_test"] = mean_squared_error(y_vectors, predictions)
            #scores[y_set][x_set][m + "_mse_test_test"] = mean_squared_error(y_test_test, predictions_test_test)
            
            #train-test
            models[m].fit(x_train, y_train)
            predictions = models[m].predict(x_test)
            predictions_test_test = models[m].predict(x_test_test)
            sensitivity = recall_score(y_test, predictions)
            specificity = recall_score(y_test, predictions, pos_label=0)
            sensitivity_test_test = recall_score(y_test_test, predictions_test_test)
            specificity_test_test = recall_score(y_test_test, predictions_test_test, pos_label=0)
            scores[y_set][x_set][m + "_mse_test"] = mean_squared_error(y_test, predictions)
            scores[y_set][x_set][m + "_mse_test_test"] = mean_squared_error(y_test_test, predictions_test_test)
            scores[y_set][x_set][m + "_sensitivity"] = sensitivity
            scores[y_set][x_set][m + "_specificity"] = specificity
            scores[y_set][x_set][m + "_sensitivity_test_test"] = sensitivity_test_test
            scores[y_set][x_set][m + "_specificity_test_test"] = specificity_test_test

            print(counter)
            
            if y_set == 'PKM2_inhibition':
                if sensitivity_test_test> best_sensitivity_PKM:
                    best_sensitivity_PKM=sensitivity_test_test
                    best_model_PKM=copy.deepcopy(models[m])
                    best_descriptors_ERK=copy.deepcopy(x_set)

            if y_set == 'ERK2_inhibition':        
                if sensitivity_test_test> best_sensitivity_ERK:
                    best_sensitivity_ERK=sensitivity_test_test
                    best_model_ERK=copy.deepcopy(models[m])
                    best_descriptors_ERK=copy.deepcopy(x_set)


                
print('Klaar')
print(best_model_ERK)
print(best_model_PKM)

#%%
# Convert nested dictionary to a list of records
records = []
for y_set, x_set_dict in scores.items():
    for x_set, model_scores in x_set_dict.items():
        record = {'y_set': y_set, 'x_set': x_set}
        record.update(model_scores)
        records.append(record)

# Create DataFrame from records
scores_df = pd.DataFrame(records)

# Pivot the DataFrame to get x_set as rows and the rest as columns
scores_df = scores_df.pivot(index='x_set', columns='y_set')

# Flatten the MultiIndex columns
scores_df.columns = ['_'.join(col).strip() for col in scores_df.columns.values]
mse_df = scores_df[[col for col in scores_df.columns if 'mse_test' in col]]
mse_test_test_df = scores_df[[col for col in scores_df.columns if 'mse_test_test' in col]]
sensitivity_df = scores_df[[col for col in scores_df.columns if 'sensitivity' in col]]
specificity_df = scores_df[[col for col in scores_df.columns if 'specificity' in col]]
sensitivity_test_test_df = scores_df[[col for col in scores_df.columns if 'sensitivity_test_test' in col]]
specificity_test_test_df = scores_df[[col for col in scores_df.columns if 'specificity_test_test' in col]]



#%%

scores_df


# %%

untested_molecules = pd.read_csv('untested_molecules.csv')
untested_molecules['rdkit_mol'] = untested_molecules['SMILES'].apply(lambda x: rdkit.Chem.MolFromSmiles(x))
untested_molecules['ECFP4'] = untested_molecules['rdkit_mol'].apply(lambda x: list(AllChem.GetMorganFingerprintAsBitVect(x, 2, nBits=2048)))
untested_molecules['ECFP6'] = untested_molecules['rdkit_mol'].apply(lambda x: list(AllChem.GetMorganFingerprintAsBitVect(x, 3, nBits=2048)))
untested_molecules['MACCS'] = untested_molecules['rdkit_mol'].apply(lambda x: list(AllChem.GetMACCSKeysFingerprint(x)))

#%%

x_vectors_ERK = np.array([np.array(x) for x in untested_molecules[best_descriptors_ERK]])
x_vectors_PKM = np.array([np.array(x) for x in untested_molecules[best_descriptors_PKM]])
print(x_vectors)
ERK_activity = best_model_ERK.predict(x_vectors_ERK)
PKM_activity = best_model_PKM.predict(x_vectors_PKM)
untested_molecules['ERK2_inhibition'] = ERK_activity
untested_molecules['PKM2_inhibition'] = PKM_activity

PKM_activite_rows = untested_molecules[untested_molecules['PKM2_inhibition'].eq(1)]
ERK_activite_rows = untested_molecules[untested_molecules['ERK2_inhibition'].eq(1)]
print(PKM_activite_rows.index)
print(ERK_activite_rows.index)
print(len(PKM_activite_rows.index))
print(len(ERK_activite_rows.index))

    


#untested_molecules.drop(columns=['rdkit_mol', 'ECFP4', 'ECFP6', 'MACCS'], inplace=True)

#untested_molecules.to_csv('untested_molecules_with_predictions.csv', index=False)


# %%
