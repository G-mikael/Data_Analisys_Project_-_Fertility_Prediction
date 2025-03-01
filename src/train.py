import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

#Making sure the program will run at the same folder the file is
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

#Reading the csv file
Fertility_df = pd.read_csv('../data/Fertility_Diagnosis.csv')
encoder = LabelEncoder()
Fertility_df["Season"] = encoder.fit_transform(Fertility_df["Season"])

# Separar modelo e treino
# Escolher o modelo
# Fazer o fit
# Ajustar parâmetros
# escolher os melhores parametros

Fertility_df_x = Fertility_df.iloc[:, :-1]
Fertility_df_y = Fertility_df.iloc[:,-1]

x_train, x_test_val, y_train, y_test_val = train_test_split(Fertility_df_x, Fertility_df_y, test_size=0.3, random_state=42)

x_val, x_test, y_val, y_test = train_test_split(x_test_val, y_test_val, test_size=0.3, random_state=42)

# Create and train decision tree
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(x_train, y_train)

# Create and train random forest
forest_model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
forest_model.fit(x_train, y_train)


y_pred_tree = tree_model.predict(x_val)
y_pred_forest = forest_model.predict(x_val)

print("Desempenho da Árvore de Decisão:")
print("Acurácia:", accuracy_score(y_val, y_pred_tree))
print(classification_report(y_val, y_pred_tree))

print("Desempenho da Random Forest:")
print("Acurácia:", accuracy_score(y_val, y_pred_forest))
print(classification_report(y_val, y_pred_forest))

    