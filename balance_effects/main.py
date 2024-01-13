import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the CSV file
data = pd.read_csv('balance_effects/resources/fetal_health.csv')
#sns.pairplot(data=data, hue="fetal_health")

# Drop irrelevant columns
data = data.drop(["severe_decelerations", "histogram_tendency"], axis=1)

# Show distribution by target variables of data
plt.figure(figsize=(8,6))
sns.countplot(x='fetal_health', data=data)
plt.title('Distribution by Target Variables')
plt.show()


# Step 2: Preprocess the data
X = data.iloc[:, :-1]  # Features
y = data.iloc[:, -1]   # Target variable

# Step 3: Load a pre-trained model or train a new one

X_train, X_test_original, y_train, y_test_original = train_test_split(X, y, test_size=0.3, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)  # Using RandomForestClassifier
model.fit(X_train, y_train)
predictions = model.predict(X_test_original)
accuracy_original = accuracy_score(y_test_original, predictions)
conf_matrix = confusion_matrix(y_test_original, predictions)
sns.heatmap(conf_matrix, annot=True, fmt='d')
plt.title(f'Confusion Matrix\nAccuracy: {accuracy_original:.2f}')
plt.show()

#Comment results and point out the importance of reducing false negatives

# Step 6: Imbalance the data and retrain the model
class_1 = data[data['fetal_health'] == 1]
class_2 = data[data['fetal_health'] == 2]
class_3 = data[data['fetal_health'] == 3]

# Reduce the size of class 2 and class 3
class_2 = class_2.sample(frac=0.65, random_state=42)
class_3 = class_3.sample(frac=0.5, random_state=42)

# Concatenate the reduced class 1 and class 2 with class 3
data_imbalanced = pd.concat([class_1, class_2, class_3])

# Split the data into features and target variable
X_imbalanced = data_imbalanced.iloc[:, :-1]  # Features
y_imbalanced = data_imbalanced.iloc[:, -1]   # Target variable

# Train the model with the imbalanced data
X_train, X_test, y_train, y_test = train_test_split(X_imbalanced, y_imbalanced, test_size=0.3, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test_original)
accuracy_imbalanced = accuracy_score(y_test_original, predictions)
conf_matrix = confusion_matrix(y_test_original, predictions)
sns.heatmap(conf_matrix, annot=True, fmt='d')
plt.title(f'Confusion Matrix with imbalanced data\nAccuracy: {accuracy_imbalanced:.2f}')
plt.show()

# Repeat the process with different imbalance ratios
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
accuracy_ratio_v = []
ratios = [0.4, 0.3, 0.2, 0.1]
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

random_stateV = [20, 40, 60, 80]
accuracy_ratioV = []

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
        ax = axs[i, j]
        sns.heatmap(conf_matrix, annot=True, fmt='d', ax=ax)
        ax.set_title(f'CM con {ratio*100}% clase 2 y 3, RS {random_state}')
        ax.text(0.5, -0.1, f'Precisión: {accuracy_ratio:.2f}%', size=12, ha="center", transform=ax.transAxes)

plt.tight_layout()
plt.show()

ratios_percent = [ratio * 100 for ratio in ratios]
accuracy_percentages = [accuracy_ratio_v * 100 for accuracy_ratio_v in accuracy_ratio_v]
accuracy_original_percentage = accuracy_original * 100

plt.figure(figsize=(8, 6))
plt.plot(ratios_percent, accuracy_percentages, marker='o', label='Precisión con Datos Desbalanceados')
# Añadir la línea de precisión original
plt.axhline(y=accuracy_original_percentage, color='r', linestyle='--', label='Precisión Sin Desbalanceo')
plt.title('Precisión en función del porcentaje de desbalance')
plt.xlabel('Porcentaje de desbalance (%)')
plt.ylabel('Precisión (%)')
plt.xticks(ratios_percent)  # Asegura que solo se muestren los porcentajes de ratios en el eje x
plt.legend()
plt.show()


# Inicializar un diccionario para almacenar las precisiones para cada random_state
accuracy_ratios_by_random_state = {rs: [] for rs in random_stateV}

# Iterar sobre cada combinación de ratio y random_state para calcular la precisión
for ratio in ratios:
    for random_state in random_stateV:
        class_2 = data[data['fetal_health'] == 2].sample(frac=ratio, random_state=random_state)
        class_3 = data[data['fetal_health'] == 3].sample(frac=ratio, random_state=random_state)
        data_imbalanced = pd.concat([class_1, class_2, class_3])
        X_imbalanced = data_imbalanced.iloc[:, :-1]  # Features
        y_imbalanced = data_imbalanced.iloc[:, -1]   # Target variable
        X_train, X_test, y_train, y_test = train_test_split(X_imbalanced, y_imbalanced, test_size=0.2, random_state=random_state)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test_original)
        accuracy_ratio = accuracy_score(y_test_original, predictions)
        accuracy_ratios_by_random_state[random_state].append(accuracy_ratio)

# Convertir los ratios a porcentajes para el eje x
ratios_percent = [ratio * 100 for ratio in ratios]

# Graficar las líneas para cada random_state
plt.figure(figsize=(10, 6))
for random_state, accuracies in accuracy_ratios_by_random_state.items():
    # Convertir las precisiones a porcentajes
    accuracies_percent = [accuracy * 100 for accuracy in accuracies]
    plt.plot(ratios_percent, accuracies_percent, marker='o', label=f'RS {random_state}')

# Añadir detalles al gráfico
plt.title('Precisión en función del ratio de desbalance para diferentes random_state')
plt.xlabel('Ratio de desbalance (%)')
plt.ylabel('Precisión (%)')
plt.xticks(ratios_percent)
plt.legend()
plt.grid(True)
plt.show()
# Posibilidad de cambiar el random_state para comprobar mejor el efecto del diezmado de muestras.
# Mejorar graficado de los resultados
# Cuantificar el impacto del diezmado en la accuracy
# Regerar el dataset a partir del conjunto de datos diezmado ya sea duplicando muestras o con una LLM
# Explorar otro dataset si da tiempo