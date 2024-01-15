import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# Cargar CSV
data = pd.read_csv('resources/fetal_health.csv')
#sns.pairplot(data=data, hue="fetal_health") to see the correlation between the features

# Eliminar columnas que no aportan mucha información siguiendo un análisis visual de la correlación entre las variables
data = data.drop(["severe_decelerations", "histogram_tendency"], axis=1)

# Graficar disribución de los datos
plt.figure(figsize=(8,6))
sns.countplot(x='fetal_health', data=data)
plt.title('Distribution by Target Variables')
plt.show()


# Preporcesamiento de los datos
X = data.iloc[:, :-1]  # Features
y = data.iloc[:, -1]   # Target variable

# Utilizar modelo de Random Forest

X_train, X_test_original, y_train, y_test_original = train_test_split(X, y, test_size=0.3, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)  #RandomForestClassifier
model.fit(X_train, y_train)
predictions = model.predict(X_test_original)
accuracy_original = accuracy_score(y_test_original, predictions)
conf_matrix = confusion_matrix(y_test_original, predictions)
sns.heatmap(conf_matrix, annot=True, fmt='d')
plt.title(f'Confusion Matrix\nAccuracy: {accuracy_original:.2f}')
plt.show()


# Separar los datos por clase
class_1 = data[data['fetal_health'] == 1]
class_2 = data[data['fetal_health'] == 2]
class_3 = data[data['fetal_health'] == 3]

# Reducir la cantidad de datos de las clases 2 y 3
class_2 = class_2.sample(frac=0.65, random_state=42)
class_3 = class_3.sample(frac=0.5, random_state=42)

# Concatenar los datos reducidos
data_imbalanced = pd.concat([class_1, class_2, class_3])

# Separar los datos en X e y
X_imbalanced = data_imbalanced.iloc[:, :-1]  # Features
y_imbalanced = data_imbalanced.iloc[:, -1]   # Target variable

# Entrener el modelo con los datos desbalanceados
X_train, X_test, y_train, y_test = train_test_split(X_imbalanced, y_imbalanced, test_size=0.3, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test_original)
accuracy_imbalanced = accuracy_score(y_test_original, predictions)
conf_matrix = confusion_matrix(y_test_original, predictions)
sns.heatmap(conf_matrix, annot=True, fmt='d')
plt.title(f'Confusion Matrix with imbalanced data\nAccuracy: {accuracy_imbalanced:.2f}')
plt.show()

# Repetir el proceso de reducción  con varios ratios
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
accuracy_ratio_v = []
ratios = [0.4, 0.3, 0.2, 0.1]

######## En las siguientes lineas de codigo la intención es generar un desbalanceo de las clases 2 y 3
##### Estas clases son las que contienen menor cantidad de datos, razón por la cual se desea evaluar la variación
#### En la precisión con un data set totalmente desbalanceado
#### Esto se hace con un random_state estatico de 42, todo esto se grafica en una matriz de confusión para cada porcentaje de datos usados

for i, ratio in enumerate(ratios):
    class_2 = data[data['fetal_health'] == 2].sample(frac=ratio, random_state=42)
    class_3 = data[data['fetal_health'] == 3].sample(frac=ratio, random_state=42)
    data_imbalanced = pd.concat([class_1, class_2, class_3])
    X_imbalanced = data_imbalanced.iloc[:, :-1]  # Features
    y_imbalanced = data_imbalanced.iloc[:, -1]   # Target variable
    X_train, X_test, y_train, y_test = train_test_split(X_imbalanced, y_imbalanced, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test_original)
    accuracy_ratio = accuracy_score(y_test_original, predictions)
    conf_matrix = confusion_matrix(y_test_original, predictions)
    accuracy_ratio_v.append(accuracy_ratio)
    ax = axs[i//2, i%2]
    sns.heatmap(conf_matrix, annot=True, fmt='d', ax=ax)
    ax.set_title(f'Confusion Matrix with {ratio*100}% of class 2 and class 3')
    ax.text(0.5, -0.1, f'Accuracy: {accuracy_ratio:.2f}', size=12, ha="center", transform=ax.transAxes)
plt.tight_layout()
plt.show()

##############################

ratios_percent = [ratio * 100 for ratio in ratios]
accuracy_percentages = [accuracy_ratio_v * 100 for accuracy_ratio_v in accuracy_ratio_v]
accuracy_original_percentage = accuracy_original * 100

############ En esta grafica se quiere observar la variación de la precisión entre un entrenamiento pseudobalanceado (original)
##### Y un dataset desbalanceado completamente

plt.figure(figsize=(8, 6))
plt.plot(ratios_percent, accuracy_percentages, marker='o', label='Precisión con Datos Desbalanceados')
plt.axhline(y=accuracy_original_percentage, color='r', linestyle='--', label='Precisión Sin Desbalanceo')
plt.title('Precisión en función del porcentaje de datos usados')
plt.xlabel('Porcentaje de datos usados (%)')
plt.ylabel('Precisión (%)')
plt.xticks(ratios_percent)  # Asegura que solo se muestren los porcentajes de ratios en el eje x
plt.legend()
plt.show()
############
###############


##### En las siguientes lineas de codigo se pretende ver la diferencia en la precisión variando los random state y el porcentaje de datos empleados

random_stateV = [20, 40, 60, 80]
accuracy_ratioV = []

# Inicializar un diccionario para almacenar las precisiones para cada random_state
accuracy_ratios_by_random_state = {rs: [] for rs in random_stateV}

fig, axs = plt.subplots(len(ratios), len(random_stateV), figsize=(15, 15))

for i, ratio in enumerate(ratios):
    for j, random_state in enumerate(random_stateV):
        class_2 = data[data['fetal_health'] == 2].sample(frac=ratio, random_state=random_state)
        class_3 = data[data['fetal_health'] == 3].sample(frac=ratio, random_state=random_state)
        data_imbalanced = pd.concat([class_1, class_2, class_3])
        X_imbalanced = data_imbalanced.iloc[:, :-1]  # Features
        y_imbalanced = data_imbalanced.iloc[:, -1]   # Target variable
        X_train, X_test, y_train, y_test = train_test_split(X_imbalanced, y_imbalanced, test_size=0.2, random_state=random_state)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test_original)
        accuracy_ratio = accuracy_score(y_test_original, predictions)
        conf_matrix = confusion_matrix(y_test_original, predictions)
        accuracy_ratioV.append(accuracy_ratio)
        accuracy_ratios_by_random_state[random_state].append(accuracy_ratio)
        ax = axs[i, j]
        sns.heatmap(conf_matrix, annot=True, fmt='d', ax=ax)
        ax.set_title(f'CM con {ratio*100}% clase 2 y 3, RS {random_state}')
        ax.text(0.5, -0.1, f'Precisión: {accuracy_ratio:.2f}%', size=12, ha="center", transform=ax.transAxes)

plt.tight_layout()
plt.show()

# Graficar las líneas para cada random_state
plt.figure(figsize=(10, 6))
for random_state, accuracies in accuracy_ratios_by_random_state.items():
    accuracies_percent = [accuracy * 100 for accuracy in accuracies]
    plt.plot(ratios_percent, accuracies_percent, marker='o', label=f'RS {random_state}')

plt.title('Precisión en función del porcentaje de datos usados para diferentes random_state')
plt.xlabel('Porcentaje de datos usados (%)')
plt.ylabel('Precisión (%)')
plt.xticks(ratios_percent)
plt.legend()
plt.grid(True)
plt.show()

### Como es posible observar la reducción de muestras para las categorias minoritarias impacta en la precision del modelo
### La elección del random state con ratios muy pequeños puede tener un impacto no despreciable en la precisión del modelo.
### La naturaleza aleatoria en la eleccion de las muestra puede llevar a elecciones representativas o no de las muestras
### Esta eleccion del random state deja de tener importancia a ratios mayores
### Aun partiendo de un dataset pseudo balanceado podemos considerar la prediccion del modelo como aceptable al alcanzar una precision del 95%
### En el mejor modelo hay 18 casos + 2 casos + 1 casos falsos negativos, que sería recomendable evitar