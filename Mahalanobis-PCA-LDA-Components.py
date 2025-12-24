import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import mahalanobis
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


# Función para la clasificación basada en la distancia de Mahalanobis
def mahalanobis_distance_classification(X_train, y_train, X_test, threshold=None):
    y_pred = []
    confidences = []

    # Verificar las dimensiones de X_train
    if X_train.ndim != 2:
        raise ValueError(f'X_train debe ser un array 2D, pero tiene {X_train.ndim} dimensiones.')

    if X_train.shape[0] < 2 or X_train.shape[1] < 2:
        raise ValueError(
            f'X_train debe tener al menos dos muestras y dos características, pero tiene {X_train.shape[0]} muestras y {X_train.shape[1]} características.')

    if len(X_train) == 0:
        raise ValueError('X_train está vacío.')

    # Calcular la matriz de covarianza
    cov_matrix = np.cov(X_train, rowvar=False)
    print(f'cov_matrix: {cov_matrix}')
    print(f'Dimensiones de cov_matrix: {cov_matrix.shape}')

    # Verificar que la matriz de covarianza sea cuadrada
    if cov_matrix.shape[0] != cov_matrix.shape[1]:
        raise ValueError(f'La matriz de covarianza no es cuadrada: {cov_matrix.shape}')

    # Intentar calcular la inversa de la matriz de covarianza
    try:
        cov_matrix_inv = np.linalg.inv(cov_matrix)
    except np.linalg.LinAlgError:
        print("Error al calcular la inversa de la matriz de covarianza. Puede ser singular.")
        cov_matrix_inv = np.linalg.inv(cov_matrix + np.eye(cov_matrix.shape[0]) * 1e-10)

    for test_point in X_test:
        distances = []
        for train_point in X_train:
            distance = mahalanobis(test_point, train_point, cov_matrix_inv)
            distances.append(distance)

        min_index = np.argmin(distances)
        min_distance = distances[min_index]
        if threshold is not None and min_distance > threshold:
            y_pred.append(None)  # Clasificación desconocida si la distancia está por encima del umbral
        else:
            y_pred.append(y_train[min_index])
        confidences.append(min_distance)
    return y_pred, confidences


# Directorio base que contiene las carpetas train_20, test_20, val_20
base_directory = "/home/usuario/Descargas/YALE/yale.v1i.folder"

# Cargar el modelo VGG16 preentrenado sin las capas de clasificación
base_model = load_model("vgg_featureX_10_model_trained.h5")

# Extraer características y etiquetas del conjunto de entrenamiento
train_directory = os.path.join(base_directory, "train")
X_train, y_train = extract_features_and_labels(train_directory, base_model)
print(f'Dimensiones de X_train: {X_train.shape}')  # Verificar dimensiones

# Extraer características y etiquetas del conjunto de prueba
test_directory = os.path.join(base_directory, "valid")
X_test, y_test = extract_features_and_labels(test_directory, base_model)
print(f'Dimensiones de X_test: {X_test.shape}')  # Verificar dimensiones

# Estandarizar características
scaler = StandardScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)

# Definir el rango de componentes a probar
component_numbers = list(range(2, 101))  # Cambiado a mínimo 2 para evitar dimensiones insuficientes
pca_accuracies = []
explained_variances = []

for num_components in component_numbers:
    # Aplicar PCA
    pca = PCA(n_components=num_components)
    pca_X_train = pca.fit_transform(scaled_X_train)
    pca_X_test = pca.transform(scaled_X_test)

    # Obtener la varianza explicada para el número de componentes actual
    explained_variance = pca.explained_variance_ratio_
    explained_variances.append(
        np.sum(explained_variance))  # Varianza explicada acumulada para el número de componentes actual

    # Aplicar LDA
    num_lda_components = min(len(np.unique(y_train)) - 1, num_components)
    lda = LDA(n_components=num_lda_components)
    lda_X_train = lda.fit_transform(pca_X_train, y_train)
    lda_X_test = lda.transform(pca_X_test)

    # Verificar dimensiones de los datos transformados
    print(f'Dimensiones de lda_X_train: {lda_X_train.shape}')
    print(f'Dimensiones de lda_X_test: {lda_X_test.shape}')

    if lda_X_train.shape[1] < 2:
        print(f'Skipping LDA with {num_components} PCA components due to insufficient dimensions.')
        continue

    # Realizar predicciones en el conjunto de prueba con la distancia de Mahalanobis
    threshold = 1000  # Ajustar según sea necesario
    y_pred_mahalanobis, confidences_mahalanobis = mahalanobis_distance_classification(lda_X_train, y_train, lda_X_test,
                                                                                      threshold)

    # Filtrar predicciones None (umbral no alcanzado)
    y_pred_mahalanobis_filtered = [pred for pred in y_pred_mahalanobis if pred is not None]
    y_test_filtered = [true for pred, true in zip(y_pred_mahalanobis, y_test) if pred is not None]

    # Evaluar las predicciones de la distancia de Mahalanobis
    accuracy_mahalanobis = accuracy_score(y_test_filtered, y_pred_mahalanobis_filtered)
    pca_accuracies.append(accuracy_mahalanobis)

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
