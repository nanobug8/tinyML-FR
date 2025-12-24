import os
import numpy as np
import pickle
from sklearn.svm import SVC
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import load_model
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
num_components = 6  # Ajustar según sea necesario
pca = PCA(n_components=num_components)
pca_X_train = pca.fit_transform(scaled_X_train)
pca_X_test = pca.transform(scaled_X_test)

# Guardar el modelo PCA entrenado
with open('pca_model.pkl', 'wb') as f:
    pickle.dump(pca, f)

# Aplicar LDA
num_classes = len(np.unique(y_train))
lda = LDA(n_components=min(num_classes - 1, num_components))
lda_X_train = lda.fit_transform(pca_X_train, y_train)
lda_X_test = lda.transform(pca_X_test)

# Guardar el modelo LDA entrenado
with open('lda_model.pkl', 'wb') as f:
    pickle.dump(lda, f)

# Visualizar características LDA en 2D
lda_2d = lda.transform(pca_X_train)[:, :2]
df = pd.DataFrame({
    'LDA1': lda_2d[:, 0],
    'LDA2': lda_2d[:, 1],
    'label': y_train
})

plt.figure(figsize=(10, 8))
sns.scatterplot(
    x='LDA1', y='LDA2',
    hue='label',
    palette=sns.color_palette('hsv', len(np.unique(y_train))),
    data=df,
    legend='full',
    alpha=0.6
)
plt.title('LDA de características de imágenes (Conjunto de entrenamiento)')
plt.show()

# Entrenar SVM
svm = SVC(kernel='linear', C=1.0, probability=True)
svm.fit(lda_X_train, y_train)

# Realizar predicciones en el conjunto de prueba con SVM
y_pred_svm = svm.predict(lda_X_test)
confidences_svm = svm.predict_proba(lda_X_test).max(axis=1)

# Evaluar las predicciones de SVM
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f'Accuracy on test set (SVM): {accuracy_svm:.2f}')

print('Classification Report (SVM):')
print(classification_report(y_test, y_pred_svm))

print('Confusion Matrix (SVM):')
conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)
print(conf_matrix_svm)

# Visualizar la matriz de confusión
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix_svm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.title('Confusion Matrix (SVM)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Guardar resultados en un archivo CSV
results_df_svm = pd.DataFrame({
    'True_Label': y_test,
    'Predicted_Label_SVM': y_pred_svm,
    'Confidence_SVM': confidences_svm
})
results_df_svm.to_csv('predictions_results_svm.csv', index=False)
