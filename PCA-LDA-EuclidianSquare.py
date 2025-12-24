from tensorflow.keras.models import load_model
import os
import numpy as np
import pickle
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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

# Función de clasificación basada en la distancia euclidiana al cuadrado
def euclidean_distance_squared_classification(X_train, y_train, X_test, threshold=None):
    y_pred = []
    confidences = []

    for test_point in X_test:
        # Calcula la distancia euclidiana al cuadrado
        distances = [np.sum(np.square(test_point - train_point)) for train_point in X_train]
        min_index = np.argmin(distances)
        min_distance = distances[min_index]

        # Clasificación desconocida si la distancia está por encima del umbral
        if threshold is not None and min_distance > threshold:
            y_pred.append("Unknown")
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

# Extraer características y etiquetas del conjunto de prueba
test_directory = os.path.join(base_directory, "valid")
X_test, y_test = extract_features_and_labels(test_directory, base_model)

# Estandarizar características
scaler = StandardScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)

# Guardar el scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Aplicar PCA
num_components = 26  # Ajustar según sea necesario
pca = PCA(n_components=num_components)
pca_X_train = pca.fit_transform(scaled_X_train)
pca_X_test = pca.transform(scaled_X_test)

# Guardar el modelo PCA entrenado
with open('pca_model.pkl', 'wb') as f:
    pickle.dump(pca, f)

# Visualizar características PCA en 2D (opcional)
pca_2d = pca_X_train[:, :2]
df_pca = pd.DataFrame({
    'PCA1': pca_2d[:, 0],
    'PCA2': pca_2d[:, 1],
    'label': y_train
})

plt.figure(figsize=(10, 8))
sns.scatterplot(
    x='PCA1', y='PCA2',
    hue='label',
    palette=sns.color_palette('hsv', len(np.unique(y_train))),
    data=df_pca,
    legend='full',
    alpha=0.6
)
plt.title('PCA de características de imágenes (Conjunto de entrenamiento)')
plt.show()

# Aplicar LDA
lda = LDA(n_components=14)
lda_X_train = lda.fit_transform(pca_X_train, y_train)
lda_X_test = lda.transform(pca_X_test)

# Guardar el modelo LDA entrenado
with open('lda_model.pkl', 'wb') as f:
    pickle.dump(lda, f)

# Visualizar características LDA en 2D
lda_2d = lda.transform(pca_X_train)[:, :2]
df_lda = pd.DataFrame({
    'LDA1': lda_2d[:, 0],
    'LDA2': lda_2d[:, 1],
    'label': y_train
})

plt.figure(figsize=(10, 8))
sns.scatterplot(
    x='LDA1', y='LDA2',
    hue='label',
    palette=sns.color_palette('hsv', len(np.unique(y_train))),
    data=df_lda,
    legend='full',
    alpha=0.6
)
plt.title('LDA de características de imágenes (Conjunto de entrenamiento)')
plt.show()

# Estandarizar características LDA
scaler_lda = StandardScaler()
scaled_lda_X_train = scaler_lda.fit_transform(lda_X_train)
scaled_lda_X_test = scaler_lda.transform(lda_X_test)

# Evaluar el rendimiento con distancia euclidiana al cuadrado
threshold = 0.5  # Ajustar según sea necesario
y_pred, confidences = euclidean_distance_squared_classification(scaled_lda_X_train, y_train, scaled_lda_X_test, threshold)

# Filtrar las predicciones desconocidas para la evaluación
valid_indices = [i for i, pred in enumerate(y_pred) if pred != "Unknown"]
y_test_filtered = [y_test[i] for i in valid_indices]
y_pred_filtered = [y_pred[i] for i in valid_indices]

# Evaluar precisión solo con predicciones válidas
accuracy = accuracy_score(y_test_filtered, y_pred_filtered)
print(f'Accuracy on test set (Euclidean Distance Squared): {accuracy:.2f}')

print('Classification Report:')
print(classification_report(y_test_filtered, y_pred_filtered))

print('Confusion Matrix:')
conf_matrix = confusion_matrix(y_test_filtered, y_pred_filtered)
print(conf_matrix)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test_filtered), yticklabels=np.unique(y_test_filtered))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (Euclidean Distance Squared)')
plt.show()

# Guardar resultados en un archivo CSV
results_df = pd.DataFrame({
    'True_Label': y_test,
    'Predicted_Label': y_pred,
    'Confidence': confidences
})
results_df.to_csv('predictions_results_euclidean_squared.csv', index=False)
