import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
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
pca_component_numbers = list(range(2, 101))  # Cambiado a mínimo 2 para evitar dimensiones insuficientes
lda_component_numbers = list(range(1, len(np.unique(y_train))))  # LDA componentes basado en el número de clases
pca_accuracies = []
lda_accuracies = []
explained_variances = []

for num_pca_components in pca_component_numbers:
    # Aplicar PCA
    pca = PCA(n_components=num_pca_components)
    pca_X_train = pca.fit_transform(scaled_X_train)
    pca_X_test = pca.transform(scaled_X_test)

    # Obtener la varianza explicada para el número de componentes actual
    explained_variance = pca.explained_variance_ratio_
    explained_variances.append(np.sum(explained_variance))  # Varianza explicada acumulada para el número de componentes actual

    # Probar diferentes números de componentes para LDA
    for num_lda_components in lda_component_numbers:
        # Aplicar LDA
        lda = LDA(n_components=num_lda_components)
        try:
            lda_X_train = lda.fit_transform(pca_X_train, y_train)
            lda_X_test = lda.transform(pca_X_test)
        except ValueError:
            continue  # Si LDA falla por alguna razón, simplemente continúa con el siguiente número de componentes

        # Verificar dimensiones de los datos transformados
        if lda_X_train.shape[1] < 2:
            continue  # LDA no es aplicable si el número de componentes es menor a 2

        # Entrenar el clasificador SVM
        svm = SVC(kernel='linear')
        svm.fit(lda_X_train, y_train)

        # Realizar predicciones en el conjunto de prueba
        y_pred_svm = svm.predict(lda_X_test)

        # Evaluar las predicciones del clasificador SVM
        accuracy_svm = accuracy_score(y_test, y_pred_svm)
        lda_accuracies.append((num_pca_components, num_lda_components, accuracy_svm))

# Crear DataFrame para resultados
lda_results_df = pd.DataFrame(lda_accuracies, columns=['PCA_Components', 'LDA_Components', 'Accuracy'])

# Guardar resultados en un archivo CSV
lda_results_df.to_csv('pca_lda_evaluation_results.csv', index=False)

# Graficar varianza explicada acumulada vs. Número de Componentes PCA
plt.figure(figsize=(10, 6))
plt.plot(pca_component_numbers, explained_variances, marker='o', linestyle='-', color='b')
plt.xlabel('Número de Componentes PCA')
plt.ylabel('Varianza Explicada Acumulada')
plt.title('Varianza Explicada Acumulada vs. Número de Componentes PCA')
plt.grid(True)
plt.show()

# Graficar precisión vs. Número de Componentes PCA y LDA
plt.figure(figsize=(10, 6))
for num_pca_components in pca_component_numbers:
    subset = lda_results_df[lda_results_df['PCA_Components'] == num_pca_components]
    plt.plot(subset['LDA_Components'], subset['Accuracy'], marker='o', linestyle='-', label=f'PCA: {num_pca_components}')

plt.xlabel('Número de Componentes LDA')
plt.ylabel('Precisión')
plt.title('Precisión vs. Número de Componentes PCA y LDA')
plt.legend(title='Número de Componentes PCA')
plt.grid(True)
plt.show()

# Encontrar el mejor número de componentes PCA y LDA
best_result = lda_results_df.loc[lda_results_df['Accuracy'].idxmax()]
best_pca_components = best_result['PCA_Components']
best_lda_components = best_result['LDA_Components']
best_accuracy = best_result['Accuracy']
print(f"Mejor número de componentes PCA: {best_pca_components}")
print(f"Mejor número de componentes LDA: {best_lda_components}")
print(f"Precisión con {best_pca_components} componentes PCA y {best_lda_components} componentes LDA: {best_accuracy:.2f}")
