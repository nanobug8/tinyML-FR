import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense
from tensorflow_model_optimization.sparsity import keras as sparsity
import numpy as np

# Ruta al modelo original
model_path = 'model.h5'

# Cargar el modelo original
model = load_model(model_path)


# 1. Cuantización de Post-Entrenamiento

def convert_to_tflite_quantized(model, output_path):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model_quantized = converter.convert()
    with open(output_path, 'wb') as f:
        f.write(tflite_model_quantized)
    print(f'Modelo cuantizado guardado en {output_path}')


convert_to_tflite_quantized(model, 'model_quantized.tflite')


# 2. Poda de Modelo

def prune_and_convert_to_tflite(model, output_path, x_train, y_train):
    # Configuración de poda
    pruning_params = {
        'pruning_schedule': sparsity.PolynomialDecay(
            initial_sparsity=0.0,
            final_sparsity=0.5,
            begin_step=0,
            end_step=1000
        )
    }

    # Aplicar poda al modelo
    model_for_pruning = sparsity.prune_low_magnitude(model, **pruning_params)

    # Compilar el modelo para entrenamiento
    model_for_pruning.compile(optimizer='adam',
                              loss='sparse_categorical_crossentropy',
                              metrics=['accuracy'])

    # Entrenar el modelo podado (ajusta según sea necesario)
    model_for_pruning.fit(x_train, y_train, epochs=5, steps_per_epoch=100)  # Ajusta según sea necesario

    # Finalizar la poda y guardar el modelo
    model_for_pruning = sparsity.strip_pruning(model_for_pruning)
    model_for_pruning.save('model_pruned.h5')

    # Convertir el modelo podado a TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_keras_model(model_for_pruning)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model_pruned_quantized = converter.convert()

    with open(output_path, 'wb') as f:
        f.write(tflite_model_pruned_quantized)
    print(f'Modelo podado y cuantizado guardado en {output_path}')


# Ejemplo de datos de entrenamiento (ajusta según tus datos)
# x_train, y_train = ...

# prune_and_convert_to_tflite(model, 'model_pruned_quantized.tflite', x_train, y_train)

# 3. Distilación de Modelos

def distill_and_convert_to_tflite(original_model, output_path, x_train, y_train, input_shape, num_classes):
    # Crear un modelo más pequeño (estudiante)
    student_model = Sequential([
        Dense(128, activation='relu', input_shape=(input_shape,)),
        Dense(num_classes, activation='softmax')
    ])

    # Compilar el modelo estudiante
    student_model.compile(optimizer='adam',
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])

    # Entrenar el modelo estudiante (ajusta según sea necesario)
    student_model.fit(x_train, y_train, epochs=5)  # Ajusta según sea necesario

    # Convertir el modelo estudiante a TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_keras_model(student_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model_student = converter.convert()

    with open(output_path, 'wb') as f:
        f.write(tflite_model_student)
    print(f'Modelo estudiante guardado en {output_path}')

# Ejemplo de datos de entrenamiento (ajusta según tus datos)
# x_train, y_train = ...
# input_shape = x_train.shape[1]
# num_classes = len(np.unique(y_train))

# distill_and_convert_to_tflite(model, 'student_model.tflite', x_train, y_train, input_shape, num_classes)
