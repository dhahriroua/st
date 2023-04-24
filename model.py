import pandas as pd
import numpy as np
import joblib
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv('data/dataset_sans_type.csv' , low_memory=False )
print(data.head())
print(data['Class'].unique())
print(data.columns)

data = data.drop([ 'Active Max', 'Active Min','Idle Max',
       'Idle Min','Unnamed: 0', 'Fwd Packet Length Max',
       'Fwd Packet Length Min', 'Fwd Packet Length Mean',
       'Fwd Packet Length Std', 'Bwd Packet Length Max',
       'Bwd Packet Length Min', 'Bwd Packet Length Mean',
       'Bwd Packet Length Std', 
        'Flow IAT Max', 'Flow IAT Min',
         'Fwd IAT Std', 'Fwd IAT Max',
       'Fwd IAT Min',  'Bwd IAT Mean', 
       'Bwd IAT Max', 'Bwd IAT Min',
       'Fwd Header Length',
       'Bwd Header Length',
       'Packet Length Min', 'Packet Length Max',
       'Packet Length Std', 'Packet Length Variance' ,
                  'Fwd Packets/s', 'Bwd Packets/s','Fwd Seg Size Min', ], axis=1)


print(data.columns)

data.shape
X = data.drop('Class', axis=1)
y = data['Class']
print(X.head(2))
print(y.head(2))

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3)
print(X_train.shape)
print(y_test.shape)



# Creating labelEncoder
le = preprocessing.LabelEncoder()
# Converting string labels into numbers.
data['Class']=le.fit_transform(data['Class'])



# Importer les bibliothèques nécessaires




# Diviser les données en ensembles d'apprentissage et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normaliser les données
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialiser le classificateur MLP
mlp = MLPClassifier(hidden_layer_sizes=(300 , 150  ), max_iter=1000, alpha=0.0001, solver='adam', random_state=42)

# Entraîner le classificateur
mlp.fit(X_train, y_train)

# Prédire les étiquettes pour l'ensemble de test
y_pred = mlp.predict(X_test)

# Évaluer la performance du modèle
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")


# save the model to disk
joblib.dump(mlp, "monmodelmlp.sav")
