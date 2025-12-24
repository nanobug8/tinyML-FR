import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.regularizers import l2

def setup_data_generators(train_data_dir, valid_data_dir, img_size=(112, 112), batch_size=32):
    train_datagen = ImageDataGenerator(
        rescale=(1. / 255),
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        rotation_range=20,  # Ejemplo: rotación aleatoria de hasta 20 grados
        width_shift_range=0.2,  # Ejemplo: desplazamiento horizontal aleatorio
        height_shift_range=0.2,  # Ejemplo: desplazamiento vertical aleatorio
        brightness_range=[0.8, 1.2],  # Ejemplo: ajuste aleatorio de brillo
        fill_mode='nearest'  # Modo de llenado para píxeles fuera de los límites de la imagen
    )

    valid_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        directory=train_data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    valid_generator = valid_datagen.flow_from_directory(
        directory=valid_data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    return train_generator, valid_generator


def build_VGG16_model(input_shape=(112, 112, 3), num_classes=10, l2_reg=0.01, dropout_rate=0.5):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten()(x)
    x = Dense(4096, activation='relu', kernel_regularizer=l2(l2_reg))(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(4096, activation='relu', kernel_regularizer=l2(l2_reg))(x)
    x = Dropout(dropout_rate)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def compile_model(model, learning_rate=0.01):
    model.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, train_generator, valid_generator, epochs=120, patience=50, reduce_lr_factor=0.01, reduce_lr_patience=5, min_lr=1e-6):
    checkpoint_fine_tune = ModelCheckpoint("vgg_model_fine_tune.h5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=patience, verbose=1, mode='max', restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=reduce_lr_factor, patience=reduce_lr_patience, min_lr=min_lr, verbose=1)

    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        validation_data=valid_generator,
        validation_steps=len(valid_generator),
        callbacks=[checkpoint_fine_tune, early_stopping, reduce_lr]
    )
    return history

def save_model(model, filename):
    model.save(filename)

def load_trained_model(filename):
    return load_model(filename)

def load_and_process_image(img_path, target_size=(112, 112)):
    img = load_img(img_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def predict_image_class(model, img_path, class_indices):
    img = load_and_process_image(img_path)
    predictions = model.predict(img)
    predicted_class_index = np.argmax(predictions)
    predicted_class_name = [k for k, v in class_indices.items() if v == predicted_class_index][0]
    return predicted_class_name

def plot_training_history(history, filename='training_history.png'):
    train_acc = history.history['accuracy']
    train_loss = history.history['loss']
    val_acc = history.history['val_accuracy']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_acc) + 1), train_acc, label='Training Accuracy')
    plt.plot(range(1, len(val_acc) + 1), val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_loss) + 1), train_loss, label='Training Loss')
    plt.plot(range(1, len(val_loss) + 1), val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

def freeze_layers(model, layer_names):
    for layer in model.layers:
        if layer.name in layer_names:
            layer.trainable = False
        else:
            layer.trainable = True

def unfreeze_layers(model, layer_names):
    for layer in model.layers:
        if layer.name in layer_names:
            layer.trainable = True
        else:
            layer.trainable = False

if __name__ == "__main__":
    train_data_dir = "/home/usuario/Descargas/DigiFace1M/dataset/train_10"
    valid_data_dir = "/home/usuario/Descargas/DigiFace1M/dataset/val_10"

    # Configurar generadores de datos
    train_generator, valid_generator = setup_data_generators(train_data_dir, valid_data_dir)

    # Construir modelo VGG16
    model = build_VGG16_model()

    # Congelar capas específicas antes del entrenamiento
    freeze_layers(model, ['block1_conv1', 'block1_conv2', 'block2_conv1', 'block2_conv2','block3_conv1','block3_conv2','block3_conv3','block4_conv1','block4_conv2','block4_conv3','block5_conv1','block5_conv2'])

    # Compilar el modelo
    model = compile_model(model)

    # Entrenar el modelo
    history = train_model(model, train_generator, valid_generator, reduce_lr_factor=0.01, reduce_lr_patience=5, min_lr=1e-6)

    # Guardar modelo entrenado
    save_model(model, "Complete_model_vgg_featureX_10_trained.h5")

    # Cargar modelo entrenado y recortar en la capa 'block5_conv3'
    loaded_model = load_trained_model("Complete_model_vgg_featureX_10_trained.h5")
    cut_model = Model(inputs=loaded_model.input, outputs=loaded_model.get_layer('block5_conv3').output)

    # Resumen del modelo recortado
    cut_model.summary()

    # Guardar modelo recortado
    cut_model.save("vgg_featureX_10_model_trained.h5")

    # Graficar historial de entrenamiento
    plot_training_history(history, 'VGG-featureX_10Clases90.png')

    # Ejemplo de predicción de imagen
    image_path = "/home/usuario/Descargas/DigiFace1M/dataset/test_10/301/54.png"
    labels = train_generator.class_indices
    predicted_class_name = predict_image_class(loaded_model, image_path, labels)
    print("Predicted class name:", predicted_class_name)