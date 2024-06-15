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

TESTED_CSV = r'C:\\Users\\eigenaar\\Documents\\8CC00\\tested_molecules.csv'
KNIME_CSV = r'C:\\Users\\eigenaar\\Documents\\8CC00\\KNIME_filtered_descriptors.csv'
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


# create X_train, X_test, y_train and y_test datasets

x_sets_list = ['physiochemical_descriptors', 'counter_descriptors', 'all_descriptors', 'ECFP4', 'ECFP6', 'MACCS']
y_sets_list = ['PKM2_inhibition', 'ERK2_inhibition']
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
#from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error
#from xgboost import XGBRegressor
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
# Default models
models = {"nnet": MLPClassifier(random_state=42),
          "rf_class": RandomForestClassifier(n_estimators=100, random_state=42),
          "xgb_clf": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
          "svm": SVC(kernel='rbf', C=1.0, gamma='scale')}
        
        
          #"nnet": MLPRegressor(random_state=42)
          #"nnet": MLPRegressor(random_state=42),}
          #"rf_regress": RandomForestRegressor(n_estimators=100, random_state=42),
          #"xgb_regr": XGBRegressor(use_label_encoder=False, eval_metric='logloss', random_state=42)}
          #"nnet": MLPRegressor(random_state=42)}
          #"svr": SVR(gamma='auto')}
best_model=0
best_accuracy=0
scores = {}
model_accuracy={}
counter=0
for y_set in y_sets_list:
    for x_set in x_sets_list:
        x_vectors = np.array([np.array(x) for x in knime_filtered[x_set]])
        y_vectors = knime_filtered[y_set].values

        x_train, x_test, y_train, y_test = train_test_split(x_vectors, y_vectors, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True)
        for m in models:
            counter= counter+1
            models[m].fit(x_train, y_train)
            y_pred = models[m].predict(x_test)
            print(counter)
            print(models[m])

            print("Accuracy:", accuracy_score(y_test, y_pred))
            #print("Classification Report:\n", classification_report(y_test, y_pred))
            #scores[f][m + "_r2_test"] = r2_score(y_test, y_pred)
            scores[m + "_mse_test"] = mean_squared_error(y_test, y_pred)
            model_accuracy[m + "_accuracy"]= accuracy_score(y_test, y_pred)
            accuracy=accuracy_score(y_test, y_pred)
            if accuracy> best_accuracy:
                best_accuracy=accuracy
                best_model=counter, m
                best_pred=y_pred
print('Klaar')
print(scores)
print(model_accuracy)
print("best model:",best_model)
print("best accuracy",best_accuracy)
print(best_pred)
print(y_test)