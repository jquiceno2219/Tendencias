import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Cargar el conjunto de datos
data = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Dividir en características y etiquetas
X = pd.get_dummies(data[['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService',
                         'MultipleLines', 'InternetService', 'OnlineSecurity','OnlineBackup','DeviceProtection', 'TechSupport',
                          'StreamingTV', 'StreamingMovies','Contract','PaperlessBilling','PaymentMethod','MonthlyCharges','TotalCharges']])

y = data['Churn']

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crear el modelo KNN
knn = KNeighborsClassifier(n_neighbors=3)

# Entrenar el modelo
knn.fit(X_train, y_train)

# Nueva respuesta del estudiante
nueva_respuesta = {
    'customerID': ['9305-MKNFE'],
    'gender': ['Female'],
    'SeniorCitizen': [1],
    'Partner': ['No'],
    'Dependents': ['No'],
    'tenure': [24],
    'PhoneService': ['Yes'],
    'MultipleLines': ['Yes'],
    'InternetService': ['Fiber optic'],
    'OnlineSecurity': ['No'],
    'OnlineBackup': ['No'],
    'DeviceProtection': ['Yes'],
    'TechSupport': ['Yes'],
    'StreamingTV': ['Yes'],
    'StreamingMovies': ['No'],
    'Contract': ['Two year'],
    'PaperlessBilling': ['Yes'],
    'PaymentMethod': ['Credit card (automatic)'],
    'MonthlyCharges': [99.65],
    'TotalCharges': [2383.4],
    'Churn': ['Yes']
}

nueva_respuesta = pd.DataFrame(nueva_respuesta)
nueva_respuesta = pd.get_dummies(nueva_respuesta)

# Asegurarse de que las columnas coincidan con las del conjunto de entrenamiento
nueva_respuesta = nueva_respuesta.reindex(columns=X.columns, fill_value=0)

# Realizar predicción
prediccion = knn.predict(nueva_respuesta)
print("Churn:", prediccion[0])