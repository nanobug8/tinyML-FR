import numpy as np
import pickle
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import load_model
from scipy.spatial.distance import cosine

# Cargar modelos y datos guardados
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('pca_model.pkl', 'rb') as f:
    pca = pickle.load(f)

with open('lda_model.pkl', 'rb') as f:
    lda = pickle.load(f)

with open('lda_X_train.pkl', 'rb') as f:
    lda_X_train = pickle.load(f)

with open('y_train.pkl', 'rb') as f:
    y_train = pickle.load(f)

# Cargar el modelo VGG16 preentrenado
base_model = load_model("vgg_featureX_10_model_trained.h5")


# Función para extraer características de una imagen
def process_image(image_path, model):
    img = load_img(image_path, target_size=(112, 112))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    features = model.predict(img_array)
    return features.flatten()


# Función para predecir usando distancia del coseno
def predict_with_cosine_distance(image_path, scaler, pca, lda, lda_X_train, y_train, model, threshold=0.7):
    # Extraer características de la imagen
    features = process_image(image_path, model)
    print(f'Extracted Features: {features.shape}')  # Imprime las características extraídas

    # Preprocesar características
    scaled_features = scaler.transform([features])
    pca_features = pca.transform(scaled_features)
    lda_features = lda.transform(pca_features)

    print(f'LDA Features: {lda_features.shape}')  # Imprime las características LDA de la imagen

    # Clasificación basada en la distancia del coseno
    similarities = [1 - cosine(lda_features.flatten(), train_point) for train_point in lda_X_train]

    print(f'Similarities: {similarities}')  # Imprime las similitudes calculadas

    max_index = np.argmax(similarities)
    max_similarity = similarities[max_index]

    print(f'Max Similarity: {max_similarity}')  # Imprime la máxima similitud

    if max_similarity < threshold:
        return None  # Clasificación desconocida si la similitud está por debajo del umbral
    else:
        return y_train[max_index]


# Ruta de la imagen a predecir
image_path = "/home/usuario/Descargas/YALE/yale.v1i.folder/valid_pca/MATI/3.png"

# Realizar la predicción
predicted_label = predict_with_cosine_distance(image_path, scaler, pca, lda, lda_X_train, y_train, base_model)
print(f'Predicted Label: {predicted_label}')
