
#%%

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
#%%

# Load datasets
TESTED_CSV = r'C:\Users\Joost\Desktop\8CC00\8CC00A3\nieuwemap\tested_molecules.csv'
UNTETSED_CSV = r'C:\Users\Joost\Desktop\8CC00\8CC00A3\nieuwemap\untested_molecules-3.csv'

# Load datasets
tested_molecules = pd.read_csv(TESTED_CSV)
untested_molecules = pd.read_csv(UNTETSED_CSV)

# List of descriptors
desc_list = [n[0] for n in Descriptors._descList]
calc = MoleculeDescriptors.MolecularDescriptorCalculator(desc_list)

# Generate RDKit molecules from SMILES
tested_molecules['rdkit_mol'] = tested_molecules['SMILES'].apply(lambda x: Chem.MolFromSmiles(x))
untested_molecules['rdkit_mol'] = untested_molecules['SMILES'].apply(lambda x: Chem.MolFromSmiles(x))

# Generate descriptors
tested_molecules['Descriptors'] = tested_molecules['rdkit_mol'].apply(lambda x: calc.CalcDescriptors(x))
untested_molecules['Descriptors'] = untested_molecules['rdkit_mol'].apply(lambda x: calc.CalcDescriptors(x))


# Prepare data for modeling
X = np.array(tested_molecules['Descriptors'].tolist())
y_pkm2 = tested_molecules['PKM2_inhibition']
y_erk2 = tested_molecules['ERK2_inhibition']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets for PKM2_inhibition
X_train_pkm2, X_test_pkm2, y_train_pkm2, y_test_pkm2 = train_test_split(X_scaled, y_pkm2, test_size=0.2, random_state=42)

# Split the data into training and testing sets for ERK2_inhibition
X_train_erk2, X_test_erk2, y_train_erk2, y_test_erk2 = train_test_split(X_scaled, y_erk2, test_size=0.2, random_state=42)

# Define a neural network model
def build_model():
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(X_train_pkm2.shape[1],)),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train a neural network model for PKM2_inhibition
model_pkm2 = build_model()
history_pkm2 = model_pkm2.fit(X_train_pkm2, y_train_pkm2, epochs=100, validation_split=0.2, verbose=1, callbacks=[early_stopping])

# Train a neural network model for ERK2_inhibition
model_erk2 = build_model()
history_erk2 = model_erk2.fit(X_train_erk2, y_train_erk2, epochs=100, validation_split=0.2, verbose=1, callbacks=[early_stopping])

# Evaluate the model for PKM2_inhibition
y_pred_pkm2 = model_pkm2.predict(X_test_pkm2).flatten()
mse_pkm2 = mean_squared_error(y_test_pkm2, y_pred_pkm2)
print(f'PKM2_inhibition Mean Squared Error: {mse_pkm2}')

# Evaluate the model for ERK2_inhibition
y_pred_erk2 = model_erk2.predict(X_test_erk2).flatten()
mse_erk2 = mean_squared_error(y_test_erk2, y_pred_erk2)
print(f'ERK2_inhibition Mean Squared Error: {mse_erk2}')

# Predict properties for untested molecules
X_untested = np.array(untested_molecules['Descriptors'].tolist())
X_untested_scaled = scaler.transform(X_untested)
untested_molecules['Predicted_PKM2_inhibition'] = model_pkm2.predict(X_untested_scaled).flatten()
untested_molecules['Predicted_ERK2_inhibition'] = model_erk2.predict(X_untested_scaled).flatten()

#%%
# Save the predictions
untested_molecules[['SMILES', 'Predicted_PKM2_inhibition', 'Predicted_ERK2_inhibition']].to_csv(r'C:\Users\Joost\Desktop\8CC00\8CC00A3\nieuwemap\untested_molecules-32.csv', index=False)

above_threshold_pkm2 = np.sum(y_pred_pkm2 > 0.5)
print(f'Number of predictions above 0.5 for PKM2_inhibition: {above_threshold_pkm2}')

# Count predictions above 0.5 for ERK2_inhibition
above_threshold_erk2 = np.sum(y_pred_erk2 > 0.5)
print(f'Number of predictions above 0.5 for ERK2_inhibition: {above_threshold_erk2}')
# %%
