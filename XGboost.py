# Define and train a Random Forest model for PKM2_inhibition
rf_pkm2 = RandomForestRegressor(n_estimators=100, random_state=42)
rf_pkm2.fit(pkm2_train, pkm2_train)

# Define and train a Random Forest model for ERK2_inhibition
rf_erk2 = RandomForestRegressor(n_estimators=100, random_state=42)
rf_erk2.fit(erk2_train_erk2, erk2_train)

# Evaluate the Random Forest model for PKM2_inhibition
y_pred_pkm2_rf = rf_pkm2.predict(pkm2_test)
mse_pkm2_rf = mean_squared_error(pkm2_test, pkm2_pred_rf)
print(f'PKM2_inhibition Mean Squared Error (Random Forest): {mse_pkm2_rf}')

# Evaluate the Random Forest model for ERK2_inhibition
y_pred_erk2_rf = rf_erk2.predict(X_test_erk2)
mse_erk2_rf = mean_squared_error(y_test_erk2, y_pred_erk2_rf)
print(f'ERK2_inhibition Mean Squared Error (Random Forest): {mse_erk2_rf}')

#%%

# Define and train an XGBoost model for PKM2_inhibition
xgb_pkm2 = xgb.XGBRegressor(n_estimators=100, random_state=42)
xgb_pkm2.fit(pkm2_train, pkm2_train)

# Define and train an XGBoost model for ERK2_inhibition
xgb_erk2 = xgb.XGBRegressor(n_estimators=100, random_state=42)
xgb_erk2.fit(erk2_train, erk2_train)

# Evaluate the XGBoost model for PKM2_inhibition
y_pred_pkm2_xgb = xgb_pkm2.predict(erk2_test_pkm2)
mse_pkm2_xgb = mean_squared_error(y_test_pkm2, y_pred_pkm2_xgb)
print(f'PKM2_inhibition Mean Squared Error (XGBoost): {mse_pkm2_xgb}')

# Evaluate the XGBoost model for ERK2_inhibition
y_pred_erk2_xgb = xgb_erk2.predict(erk2_test)
mse_erk2_xgb = mean_squared_error(erk2_test, erk2_pred_xgb)
print(f'ERK2_inhibition Mean Squared Error (XGBoost): {mse_erk2_xgb}')

#%%

# Predict properties for untested molecules using Random Forest
X_untested = np.array(untested_molecules['Descriptors'].tolist())
X_untested_scaled = scaler.transform(X_untested)
untested_molecules['Predicted_PKM2_inhibition_RF'] = rf_pkm2.predict(X_untested_scaled)
untested_molecules['Predicted_ERK2_inhibition_RF'] = rf_erk2.predict(X_untested_scaled)

# Predict properties for untested molecules using XGBoost
untested_molecules['Predicted_PKM2_inhibition_XGB'] = xgb_pkm2.predict(X_untested_scaled)
untested_molecules['Predicted_ERK2_inhibition_XGB'] = xgb_erk2.predict(X_untested_scaled)

# Save the predictions
untested_molecules[['SMILES', 'Predicted_PKM2_inhibition_RF', 'Predicted_ERK2_inhibition_RF',
                    'Predicted_PKM2_inhibition_XGB', 'Predicted_ERK2_inhibition_XGB']].to_csv()
