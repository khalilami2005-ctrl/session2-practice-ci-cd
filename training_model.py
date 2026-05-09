import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Charger les données
data = pd.read_csv('training_data.csv')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Entraîner le modèle
model = LinearRegression()
model.fit(X, y)

# Sauvegarder le modèle (fichier binaire)
joblib.dump(model, 'linear_model.pkl')

# Sauvegarder les coefficients dans un fichier texte
with open('linear_model.txt', 'w') as f:
    f.write(f'Coefficients: {model.coef_}\nIntercept: {model.intercept_}\n')