#Machine learning technieken
#%%
#Random Forest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
#Als onze data er zo uit ziet en niet al als X,Y data is, dan kunnen we op deze manier de data verwerken.
#Hierdoor worden de tuples gewoon samen genomen, maar dus wel gezien als gelijken.
#   continuous_tuple            discrete_tuple            inhibited
# 0 (7.0, 150.0, ...)           (3, 2, ...)                1
# 1 (5.5, 200.0, ...)           (1, 4, ...)                0
# ...
# #Halen de tuples los uit de data set
# continuous_features = pd.DataFrame(data['continuous_tuple'].tolist(), index=data.index)
# discrete_features = pd.DataFrame(data['discrete_tuple'].tolist(), index=data.index)
# #Combineer alle features in één DataFrame
# features = pd.concat([continuous_features, discrete_features], axis=1)
# # De target kolom
# target = data['inhibited']
# X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Model initialiseren
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
#42 is hierbij een random getal, maar doordat deze vast staat is het herhaalbaar en krijg je dus elke keer de zelfde resultaten als je runt.
# Trainen van het model
rf_clf.fit(X_train, y_train)

#Testen van model
# Voorspellingen maken op de test set
y_pred_rf = rf_clf.predict(X_test)
# Evalueren van het model
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf))

#Voorspellingen maken voor de nieuwe data
#Hier weer hetzelfde als eerst. Afhankelijk hoe de data eruit ziet. Als dit weer een dataset is met de moleculen en meerdere tuples.
#Met dan new_data als de nieuwe data set
# new_continuous_features = pd.DataFrame(new_data['continuous_tuple'].tolist(), index=new_data.index)
# new_discrete_features = pd.DataFrame(new_data['discrete_tuple'].tolist(), index=new_data.index)
# new_features = pd.concat([new_continuous_features, new_discrete_features], axis=1)

# Voorspellingen maken
new_predictions = rf_clf.predict(new_features)
# Resultaten bekijken
print("Nieuwe Voorspellingen:", new_predictions)

#%%
#XGBoost 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
#Als onze data er zo uit ziet en niet al als X,Y data is, dan kunnen we op deze manier de data verwerken.
#Hierdoor worden de tuples gewoon samen genomen, maar dus wel gezien als gelijken.
#   continuous_tuple            discrete_tuple            inhibited
# 0 (7.0, 150.0, ...)           (3, 2, ...)                1
# 1 (5.5, 200.0, ...)           (1, 4, ...)                0
# ...
# #Halen de tuples los uit de data set
# continuous_features = pd.DataFrame(data['continuous_tuple'].tolist(), index=data.index)
# discrete_features = pd.DataFrame(data['discrete_tuple'].tolist(), index=data.index)
# #Combineer alle features in één DataFrame
# features = pd.concat([continuous_features, discrete_features], axis=1)
# # De target kolom
# target = data['inhibited']
# X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Model initialiseren
xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
# Trainen van het model
xgb_clf.fit(X_train, y_train)
# Voorspellingen maken op de test set
y_pred_xgb = xgb_clf.predict(X_test)

# Evalueren van het model
print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("XGBoost Classification Report:\n", classification_report(y_test, y_pred_xgb))

#Voorspellingen maken voor de nieuwe data
#Hier weer hetzelfde als eerst. Afhankelijk hoe de data eruit ziet. Als dit weer een dataset is met de moleculen en meerdere tuples.
#Met dan new_data als de nieuwe data set
# new_continuous_features = pd.DataFrame(new_data['continuous_tuple'].tolist(), index=new_data.index)
# new_discrete_features = pd.DataFrame(new_data['discrete_tuple'].tolist(), index=new_data.index)
# new_features = pd.concat([new_continuous_features, new_discrete_features], axis=1)

# Voorspellingen maken
new_predictions = xgb_clf.predict(new_features)
# Resultaten bekijken
print("Nieuwe Voorspellingen:", new_predictions)
#%%
#Neural Network
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, accuracy_score
#Als onze data er zo uit ziet en niet al als X,Y data is, dan kunnen we op deze manier de data verwerken.
#Hierdoor worden de tuples gewoon samen genomen, maar dus wel gezien als gelijken.
#   continuous_tuple            discrete_tuple            inhibited
# 0 (7.0, 150.0, ...)           (3, 2, ...)                1
# 1 (5.5, 200.0, ...)           (1, 4, ...)                0
# ...
# #Halen de tuples los uit de data set
# continuous_features = pd.DataFrame(data['continuous_tuple'].tolist(), index=data.index)
# discrete_features = pd.DataFrame(data['discrete_tuple'].tolist(), index=data.index)
# #Combineer alle features in één DataFrame
# features = pd.concat([continuous_features, discrete_features], axis=1)
# # De target kolom
# target = data['inhibited']
# X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Normaliseer de features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# als de target niet binair is moet het naar een one hot ofz
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Model initialiseren
model = Sequential()

# Lagen toevoegen
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(y_train.shape[1], activation='softmax'))  # Gebruik 'sigmoid' voor binaire classificatie

# Model compileren
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # Gebruik 'binary_crossentropy' voor binaire classificatie
# Model trainen
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Model evalueren
loss, accuracy = model.evaluate(X_test, y_test)
print("Accuracy:", accuracy)
# Gedetailleerd classificatie rapport
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)
print("Classification Report:\n", classification_report(y_true, y_pred_classes))

#Voorspellingen maken voor de nieuwe data
#Hier weer hetzelfde als eerst. Afhankelijk hoe de data eruit ziet. Als dit weer een dataset is met de moleculen en meerdere tuples.
#Met dan new_data als de nieuwe data set
# new_continuous_features = pd.DataFrame(new_data['continuous_tuple'].tolist(), index=new_data.index)
# new_discrete_features = pd.DataFrame(new_data['discrete_tuple'].tolist(), index=new_data.index)
# new_features = pd.concat([new_continuous_features, new_discrete_features], axis=1)

# Normaliseer de nieuwe features
new_features = scaler.transform(new_features)
# Voorspellingen maken
new_predictions = model.predict(new_features)
# Voor binaire classificatie
new_predictions_classes = np.argmax(new_predictions, axis=1)  # Gebruik 'np.round(new_predictions)' voor binaire classificatie
print("Nieuwe Voorspellingen:", new_predictions_classes)
#%%
#Support Vector Machines
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
#Als onze data er zo uit ziet en niet al als X,Y data is, dan kunnen we op deze manier de data verwerken.
#Hierdoor worden de tuples gewoon samen genomen, maar dus wel gezien als gelijken.
#   continuous_tuple            discrete_tuple            inhibited
# 0 (7.0, 150.0, ...)           (3, 2, ...)                1
# 1 (5.5, 200.0, ...)           (1, 4, ...)                0
# ...
# #Halen de tuples los uit de data set
# continuous_features = pd.DataFrame(data['continuous_tuple'].tolist(), index=data.index)
# discrete_features = pd.DataFrame(data['discrete_tuple'].tolist(), index=data.index)
# #Combineer alle features in één DataFrame
# features = pd.concat([continuous_features, discrete_features], axis=1)
# # De target kolom
# target = data['inhibited']
# X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Normaliseer de features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model initialiseren
svm_clf = SVC(kernel='rbf', random_state=42)
# Trainen van het model
svm_clf.fit(X_train, y_train)

# Voorspellingen maken op de test set
y_pred_svm = svm_clf.predict(X_test)
# Evalueren van het model
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print("SVM Classification Report:\n", classification_report(y_test, y_pred_svm))

#Voorspellingen maken voor de nieuwe data
#Hier weer hetzelfde als eerst. Afhankelijk hoe de data eruit ziet. Als dit weer een dataset is met de moleculen en meerdere tuples.
#Met dan new_data als de nieuwe data set
# new_continuous_features = pd.DataFrame(new_data['continuous_tuple'].tolist(), index=new_data.index)
# new_discrete_features = pd.DataFrame(new_data['discrete_tuple'].tolist(), index=new_data.index)
# new_features = pd.concat([new_continuous_features, new_discrete_features], axis=1)

# Normaliseer de nieuwe features
new_features = scaler.transform(new_features)
# Voorspellingen maken
new_predictions = svm_clf.predict(new_features)
print("Nieuwe Voorspellingen:", new_predictions)
#%%
#Relevance Vector Machine
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from skrvm import RVC
from sklearn.metrics import classification_report, accuracy_score
#Als onze data er zo uit ziet en niet al als X,Y data is, dan kunnen we op deze manier de data verwerken.
#Hierdoor worden de tuples gewoon samen genomen, maar dus wel gezien als gelijken.
#   continuous_tuple            discrete_tuple            inhibited
# 0 (7.0, 150.0, ...)           (3, 2, ...)                1
# 1 (5.5, 200.0, ...)           (1, 4, ...)                0
# ...
# #Halen de tuples los uit de data set
# continuous_features = pd.DataFrame(data['continuous_tuple'].tolist(), index=data.index)
# discrete_features = pd.DataFrame(data['discrete_tuple'].tolist(), index=data.index)
# #Combineer alle features in één DataFrame
# features = pd.concat([continuous_features, discrete_features], axis=1)
# # De target kolom
# target = data['inhibited']
# X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
# Normaliseer de features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model initialiseren
rvm_clf = RVC()
# Trainen van het model
rvm_clf.fit(X_train, y_train)

# Voorspellingen maken op de test set
y_pred_rvm = rvm_clf.predict(X_test)
# Evalueren van het model
print("RVM Accuracy:", accuracy_score(y_test, y_pred_rvm))
print("RVM Classification Report:\n", classification_report(y_test, y_pred_rvm))

#Voorspellingen maken voor de nieuwe data
#Hier weer hetzelfde als eerst. Afhankelijk hoe de data eruit ziet. Als dit weer een dataset is met de moleculen en meerdere tuples.
#Met dan new_data als de nieuwe data set
# new_continuous_features = pd.DataFrame(new_data['continuous_tuple'].tolist(), index=new_data.index)
# new_discrete_features = pd.DataFrame(new_data['discrete_tuple'].tolist(), index=new_data.index)
# new_features = pd.concat([new_continuous_features, new_discrete_features], axis=1)
# Normaliseer de nieuwe features
new_features = scaler.transform(new_features)
# Voorspellingen maken
new_predictions = rvm_clf.predict(new_features)
print("Nieuwe Voorspellingen:", new_predictions)