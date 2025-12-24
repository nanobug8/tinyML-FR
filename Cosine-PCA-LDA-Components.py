import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cosine
import pandas as pd
from tensorflow.keras.models import load_model

# Función para cargar imágenes y extraer características
def extract_features_and_labels(parent_directory, model):
    labels = sorted(os.listdir(parent_directory))
    all_features = []
    all_labels = []

    for label in labels:
        label_dir = os.path.join(parent_directory, label)
        file_list = sorted(os.listdir(label_dir))

        for fname in file_list:
            img_path = os.path.join(label_dir, fname)
            img = load_img(img_path, target_size=(112, 112))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)  # Preprocesar la imagen

            features = model.predict(img_array)
            reshaped_features = features.flatten()

            all_features.append(reshaped_features)
            all_labels.append(label)

    all_features = np.array(all_features)
    all_labels = np.array(all_labels)
    return all_features, all_labels

# Función para la clasificación basada en la distancia del coseno
def cosine_distance_classification(X_train, y_train, X_test, threshold=None):
    y_pred = []
    confidences = []

    for test_point in X_test:
        similarities = [1 - cosine(test_point, train_point) for train_point in X_train]
        max_index = np.argmax(similarities)
        max_similarity = similarities[max_index]

        if threshold is not None and max_similarity < threshold:
            y_pred.append(None)  # Clasificación desconocida si la similitud está por debajo del umbral
        else:
            y_pred.append(y_train[max_index])

        confidences.append(max_similarity)

    return y_pred, confidences

# Directorio base que contiene las carpetas train_20, test_20, val_20
base_directory = "/home/usuario/Descargas/YALE/yale.v1i.folder"

# Cargar el modelo VGG16 preentrenado sin las capas de clasificación
base_model = load_model("vgg_featureX_10_model_trained.h5")

# Extraer características y etiquetas del conjunto de entrenamiento
train_directory = os.path.join(base_directory, "train")
X_train, y_train = extract_features_and_labels(train_directory, base_model)

# Extraer características y etiquetas del conjunto de prueba
test_directory = os.path.join(base_directory, "valid")
X_test, y_test = extract_features_and_labels(test_directory, base_model)

# Estandarizar características
scaler = StandardScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)

# Definir el rango de componentes a probar
component_numbers = list(range(1, 101))
pca_accuracies = []
explained_variances = []

for num_components in component_numbers:
    # Aplicar PCA
    pca = PCA(n_components=num_components)
    pca_X_train = pca.fit_transform(scaled_X_train)
    pca_X_test = pca.transform(scaled_X_test)

    # Obtener la varianza explicada para el número de componentes actual
    explained_variance = pca.explained_variance_ratio_
    explained_variances.append(np.sum(explained_variance))  # Varianza explicada acumulada para el número de componentes actual

    # Aplicar LDA
    lda = LDA(n_components=min(len(np.unique(y_train)) - 1, num_components))
    lda_X_train = lda.fit_transform(pca_X_train, y_train)
    lda_X_test = lda.transform(pca_X_test)

    # Realizar predicciones en el conjunto de prueba con la distancia del coseno
    threshold = 0.9  # Ajustar según sea necesario
    y_pred_cosine, confidences_cosine = cosine_distance_classification(lda_X_train, y_train, lda_X_test, threshold)

    # Filtrar predicciones None (umbral no alcanzado)
    y_pred_cosine_filtered = [pred for pred in y_pred_cosine if pred is not None]
    y_test_filtered = [true for pred, true in zip(y_pred_cosine, y_test) if pred is not None]

    # Evaluar las predicciones de la distancia del coseno
    accuracy_cosine = accuracy_score(y_test_filtered, y_pred_cosine_filtered)
    pca_accuracies.append(accuracy_cosine)

# Crear DataFrame para resultados
results_df = pd.DataFrame({
    'PCA_Components': component_numbers,
    'Explained_Variance': explained_variances,
    'Accuracy': pca_accuracies
})

# Guardar resultados en un archivo CSV
results_df.to_csv('pca_evaluation_results.csv', index=False)

# Graficar varianza explicada acumulada vs. Número de Componentes PCA
plt.figure(figsize=(10, 6))
plt.plot(component_numbers, explained_variances, marker='o', linestyle='-', color='b')
plt.xlabel('Número de Componentes PCA')
plt.ylabel('Varianza Explicada Acumulada')
plt.title('Varianza Explicada Acumulada vs. Número de Componentes PCA')
plt.grid(True)
plt.show()

# Graficar precisión vs. Número de Componentes PCA
plt.figure(figsize=(10, 6))
plt.plot(component_numbers, pca_accuracies, marker='o', linestyle='-', color='r')
plt.xlabel('Número de Componentes PCA')
plt.ylabel('Precisión')
plt.title('Precisión vs. Número de Componentes PCA')
plt.grid(True)
plt.show()

# Encontrar el mejor número de componentes PCA
best_pca_index = np.argmax(pca_accuracies)
best_pca_components = component_numbers[best_pca_index]
best_pca_accuracy = pca_accuracies[best_pca_index]
print(f"Mejor número de componentes PCA: {best_pca_components}")
print(f"Precisión con {best_pca_components} componentes PCA: {best_pca_accuracy:.2f}")

# Evaluar el mejor número de componentes LDA para el número óptimo de PCA
best_pca = PCA(n_components=best_pca_components)
pca_X_train_best = best_pca.fit_transform(scaled_X_train)
pca_X_test_best = best_pca.transform(scaled_X_test)

max_lda_components = min(len(np.unique(y_train)) - 1, best_pca_components)
lda_component_numbers = list(range(1, max_lda_components + 1))
lda_accuracies = []

for num_lda_components in lda_component_numbers:
    lda = LDA(n_components=num_lda_components)
    lda_X_train = lda.fit_transform(pca_X_train_best, y_train)
    lda_X_test = lda.transform(pca_X_test_best)

    # Realizar predicciones en el conjunto de prueba con la distancia del coseno
    y_pred_cosine, confidences_cosine = cosine_distance_classification(lda_X_train, y_train, lda_X_test, threshold)

    # Filtrar predicciones None (umbral no alcanzado)
    y_pred_cosine_filtered = [pred for pred in y_pred_cosine if pred is not None]
    y_test_filtered = [true for pred, true in zip(y_pred_cosine, y_test) if pred is not None]

    # Evaluar las predicciones de la distancia del coseno
    accuracy_cosine = accuracy_score(y_test_filtered, y_pred_cosine_filtered)
    lda_accuracies.append(accuracy_cosine)

# Crear DataFrame para resultados de LDA
lda_results_df = pd.DataFrame({
    'LDA_Components': lda_component_numbers,
    'Accuracy': lda_accuracies
})

# Guardar resultados de LDA en un archivo CSV
lda_results_df.to_csv('lda_evaluation_results.csv', index=False)

# Graficar precisión vs. Número de Componentes LDA
plt.figure(figsize=(10, 6))
plt.plot(lda_component_numbers, lda_accuracies, marker='o', linestyle='-', color='g')
plt.xlabel('Número de Componentes LDA')
plt.ylabel('Precisión')
plt.title(f'Precisión vs. Número de Componentes LDA (con {best_pca_components} componentes PCA)')
plt.grid(True)
plt.show()

# Imprimir resultados de LDA
best_lda_index = np.argmax(lda_accuracies)
best_lda_components = lda_component_numbers[best_lda_index]
best_lda_accuracy = lda_accuracies[best_lda_index]
print(f"Mejor número de componentes LDA: {best_lda_components}")
print(f"Precisión con {best_lda_components} componentes LDA: {best_lda_accuracy:.2f}")
