from logging import exception
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import seaborn as sns
import openai
from openai import OpenAI
from pprint import pprint
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle
from imblearn.over_sampling import ADASYN
import numpy as np


import json

## NOTA ACLARATORIA: El código de este archivo es el mismo que el de main.py correspondiente al T2, pero con la diferencia de que a partir de la línea 68 se introdujeron las nuevas capacidades.
# Cargar CSV
data = pd.read_csv('resources/fetal_health.csv')
#sns.pairplot(data=data, hue="fetal_health") to see the correlation between the features

# Eliminar columnas que no aportan mucha información siguiendo un análisis visual de la correlación entre las variables
data = data.drop(["severe_decelerations", "histogram_tendency"], axis=1)

# Preporcesamiento de los datos
X = data.iloc[:, :-1]  # Features
y = data.iloc[:, -1]   # Target variable

# Utilizar modelo de Random Forest

X_train, X_test_original, y_train, y_test_original = train_test_split(X, y, test_size=0.3, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)  #RandomForestClassifier
model.fit(X_train, y_train)
predictions = model.predict(X_test_original)
accuracy_original = accuracy_score(y_test_original, predictions)
conf_matrix_original = confusion_matrix(y_test_original, predictions)
precision_original = precision_score(y_test_original, predictions, average='weighted')
recall_original = recall_score(y_test_original, predictions, average='weighted')
f1_original = f1_score(y_test_original, predictions, average='weighted')
roc_auc_original = roc_auc_score(y_test_original, model.predict_proba(X_test_original), multi_class='ovr')

metrics = {
    'Accuracy': accuracy_original,
    'Precision': precision_original,
    'Recall': recall_original,
    'F1 Score': f1_original,
    'ROC AUC': roc_auc_original
}

# Graficar disribución de los datos y la matriz de confusión
plt.figure(figsize=(12, 6))

# Distribución de las variables
plt.subplot(1, 2, 1)
sns.countplot(x='fetal_health', data=data)
plt.title('Distribution by Target Variables')

# Matriz de confusión
plt.subplot(1, 2, 2)
sns.heatmap(conf_matrix_original, annot=True, fmt='d')
plt.title(f'Confusion Matrix Original dataset\nAccuracy: {accuracy_original:.2f}')

plt.tight_layout()
plt.show()

# Separar los datos por clase
class_1 = data[data['fetal_health'] == 1]
class_2 = data[data['fetal_health'] == 2]
class_3 = data[data['fetal_health'] == 3]

# Definir las fracciones de datos que se van a utilizar para cada clase
fractions = [0.1, 0.2, 0.3, 0.4]

num_fractions = len(fractions)
num_metrics = 5  
bar_width = 0.15

plt.figure(figsize=(15, num_fractions * 2))

for i, frac in enumerate(fractions):
    # Reducir la cantidad de datos de las clases 2 y 3
    reduced_class_2 = class_2.sample(frac=frac, random_state=42)
    reduced_class_3 = class_3.sample(frac=frac, random_state=42)

    # Concatenar los datos reducidos
    data_imbalanced = pd.concat([class_1, reduced_class_2, reduced_class_3])

    # Dividr los datos en X e y
    X_imbalanced = data_imbalanced.iloc[:, :-1]  # Features
    y_imbalanced = data_imbalanced.iloc[:, -1]   # Target variable

    # Entrenar el modelo con los datos desbalanceados
    X_train, X_test, y_train, y_test = train_test_split(X_imbalanced, y_imbalanced, test_size=0.3, random_state=42)
    model_imbalanced = RandomForestClassifier(n_estimators=100, random_state=42)
    model_imbalanced.fit(X_train, y_train)
    predictions = model_imbalanced.predict(X_test_original)
    accuracy_imbalanced = accuracy_score(y_test_original, predictions)
    conf_matrix_imbalanced = confusion_matrix(y_test_original, predictions)
    precision_imbalanced = precision_score(y_test_original, predictions, average='weighted')
    recall_imbalanced = recall_score(y_test_original, predictions, average='weighted')
    f1_imbalanced = f1_score(y_test_original, predictions, average='weighted')
    roc_auc_imbalanced = roc_auc_score(y_test_original, model.predict_proba(X_test_original), multi_class='ovr')

    metrics_imbalanced = {
        'Accuracy': accuracy_imbalanced,
        'Precision': precision_imbalanced,
        'Recall': recall_imbalanced,
        'F1 Score': f1_imbalanced,
        'ROC AUC': roc_auc_imbalanced
    }

    # Balancear los datos utilizando SMOTE
    smote = SMOTE(random_state=42)
    X_smote, y_smote = smote.fit_resample(X_imbalanced, y_imbalanced)
    X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.3, random_state=42)
    model_smote = RandomForestClassifier(n_estimators=100, random_state=42)
    model_smote.fit(X_train, y_train)
    predictions = model_smote.predict(X_test_original)
    accuracy_smote = accuracy_score(y_test_original, predictions)
    conf_matrix = confusion_matrix(y_test_original, predictions)
    precision = precision_score(y_test_original, predictions, average='weighted')
    recall = recall_score(y_test_original, predictions, average='weighted')
    f1 = f1_score(y_test_original, predictions, average='weighted')
    roc_auc = roc_auc_score(y_test_original, model.predict_proba(X_test_original), multi_class='ovr')
    
    metrics_smote = {
        'Accuracy': accuracy_smote,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC AUC': roc_auc
    }
    
    #Balancear los datos con ADASYN
    adasyn = ADASYN(random_state=42)
    X_adasyn, y_adasyn = adasyn.fit_resample(X_imbalanced, y_imbalanced)
    X_train, X_test, y_train, y_test = train_test_split(X_adasyn, y_adasyn, test_size=0.3, random_state=42)
    model_ADASYN = RandomForestClassifier(n_estimators=100, random_state=42)
    model_ADASYN.fit(X_train, y_train)
    predictions = model_ADASYN.predict(X_test_original)
    accuracy_adasyn = accuracy_score(y_test_original, predictions)
    conf_matrix_adasyn = confusion_matrix(y_test_original, predictions)
    precision_adasyn = precision_score(y_test_original, predictions, average='weighted')
    recall_adasyn = recall_score(y_test_original, predictions, average='weighted')
    f1_adasyn = f1_score(y_test_original, predictions, average='weighted')
    roc_auc_adasyn = roc_auc_score(y_test_original, model.predict_proba(X_test_original), multi_class='ovr')

    metrics_adasyn = {
        'Accuracy': accuracy_adasyn,
        'Precision': precision_adasyn,
        'Recall': recall_adasyn,
        'F1 Score': f1_adasyn,
        'ROC AUC': roc_auc_adasyn
    }

    #Balancear los datos utilizando OpenAI (API Key necesaria)
    try:
        openai = OpenAI()
        X_openai, y_openai = openai.fit_resample(X_imbalanced, y_imbalanced)
        X_train, X_test, y_train, y_test = train_test_split(X_openai, y_openai, test_size=0.3, random_state=42)
        model_openAI = RandomForestClassifier(n_estimators=100, random_state=42)
        model_openAI.fit(X_train, y_train)
        predictions = model_openAI.predict(X_test_original)
        accuracy_openai = accuracy_score(y_test_original, predictions)
        conf_matrix_openai = confusion_matrix(y_test_original, predictions)
        precision_openai = precision_score(y_test_original, predictions, average='weighted')
        recall_openai = recall_score(y_test_original, predictions, average='weighted')
        f1_openai = f1_score(y_test_original, predictions, average='weighted')
        roc_auc_openai = roc_auc_score(y_test_original, model.predict_proba(X_test_original), multi_class='ovr')

        metrics_openai = {
            'Accuracy': accuracy_openai,
            'Precision': precision_openai,
            'Recall': recall_openai,
            'F1 Score': f1_openai,
            'ROC AUC': roc_auc_openai
        }

    except Exception as e:
        print("API Key is not valid")
    
    indices = np.arange(num_metrics)

    plt.subplot(num_fractions, 1, i + 1)

    # Graficar para cada modelo

    #Original
    plt.bar(indices, list(metrics.values()), bar_width, label='Original')
    plt.bar(indices + bar_width, list(metrics_imbalanced.values()), bar_width, label='Imbalanced')
    plt.bar(indices + bar_width * 2, list(metrics_smote.values()), bar_width, label='SMOTE')

    #ADASYN model
    plt.bar(indices + 3 * bar_width, list(metrics_adasyn.values()), bar_width, label='ADASYN')

    plt.xlabel('Metrics')
    plt.ylabel('Scores')
    plt.title(f'Metrics Comparison (frac={frac})')
    plt.xticks(indices + bar_width, ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'])
    plt.legend()
    plt.ylim(0, 1)

    #OpenAI
    try:
        plt.bar(indices + 4 * bar_width, list(metrics_openai.values()), bar_width, label='OpenAI')
    except Exception as e:
        print(f"OpenAI data not available for frac={frac}")
plt.tight_layout()
plt.show()

# En los resultados se puede observar que los datos balanceados con SMOTE y ADASYN tienen un mejor desempeño que los datos originales imbalanceados para todas las métricas, 
# excepto para el ROC AUC, donde los datos originales tienen un mejor desempeño. Esto se debe a que el ROC AUC es una métrica que no se ve afectada por el desbalanceo de los datos.
