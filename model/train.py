import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# cargar dataset
df = pd.read_csv("C:/Users/minos/Desktop/FastApi/data/diabetes.csv")

# separar variables
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# modelo
model = RandomForestClassifier()
model.fit(X_train, y_train)

# precisión rápida
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)

# guardar modelo
joblib.dump(model, "model.pkl")

print("Modelo guardado correctamente")