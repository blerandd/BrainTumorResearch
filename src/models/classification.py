import tensorflow as tf
from tensorflow.keras import layers, models

CLASS_COUNT = 3

def build_custom_cnn(input_shape=(224, 224, 1), class_count=CLASS_COUNT):
    inputs = layers.Input(shape=input_shape)
    x = inputs
    for filters in (32, 64, 128, 256):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(class_count, activation="softmax")(x)
    return models.Model(inputs, outputs, name="custom_cnn")

def build_transfer_model(architecture, input_shape=(224, 224, 3), class_count=CLASS_COUNT):
    architecture = architecture.lower()
    if architecture == "vgg16":
        base = tf.keras.applications.VGG16(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape
        )
    elif architecture == "resnet50":
        base = tf.keras.applications.ResNet50(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape
        )
    else:
        raise ValueError("architecture must be 'vgg16' or 'resnet50'")

    base.trainable = False
    inputs = layers.Input(shape=input_shape)
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(class_count, activation="softmax")(x)
    model = models.Model(inputs, outputs, name=architecture)
    return model, base

def unfreeze_top_layers(base_model, trainable_layers=30):
    base_model.trainable = True
    for layer in base_model.layers[:-trainable_layers]:
        layer.trainable = False
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
