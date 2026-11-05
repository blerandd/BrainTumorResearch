
import tensorflow as tf


def build_classification_cnn(
    input_shape=(256, 256, 1),
    num_classes=3,
    learning_rate=0.001
):
    inputs = tf.keras.Input(
        shape=input_shape,
        name="input_layer"
    )

    x = tf.keras.layers.Conv2D(
        32,
        kernel_size=(3, 3),
        padding="same",
        kernel_initializer="he_normal",
        name="conv2d"
    )(inputs)

    x = tf.keras.layers.BatchNormalization(
        name="batch_normalization"
    )(x)

    x = tf.keras.layers.Activation(
        "relu",
        name="activation"
    )(x)

    x = tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2),
        name="max_pooling2d"
    )(x)

    x = tf.keras.layers.Conv2D(
        64,
        kernel_size=(3, 3),
        padding="same",
        kernel_initializer="he_normal",
        name="conv2d_1"
    )(x)

    x = tf.keras.layers.BatchNormalization(
        name="batch_normalization_1"
    )(x)

    x = tf.keras.layers.Activation(
        "relu",
        name="activation_1"
    )(x)

    x = tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2),
        name="max_pooling2d_1"
    )(x)

    x = tf.keras.layers.Conv2D(
        128,
        kernel_size=(3, 3),
        padding="same",
        kernel_initializer="he_normal",
        name="conv2d_2"
    )(x)

    x = tf.keras.layers.BatchNormalization(
        name="batch_normalization_2"
    )(x)

    x = tf.keras.layers.Activation(
        "relu",
        name="activation_2"
    )(x)

    x = tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2),
        name="max_pooling2d_2"
    )(x)

    x = tf.keras.layers.Conv2D(
        256,
        kernel_size=(3, 3),
        padding="same",
        kernel_initializer="he_normal",
        name="conv2d_3"
    )(x)

    x = tf.keras.layers.BatchNormalization(
        name="batch_normalization_3"
    )(x)

    x = tf.keras.layers.Activation(
        "relu",
        name="activation_3"
    )(x)

    x = tf.keras.layers.GlobalAveragePooling2D(
        name="global_average_pooling2d"
    )(x)

    x = tf.keras.layers.Dense(
        128,
        activation="relu",
        kernel_initializer="he_normal",
        name="dense"
    )(x)

    x = tf.keras.layers.Dropout(
        0.5,
        name="dropout"
    )(x)

    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        name="predictions"
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="Custom_Brain_Tumor_CNN"
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    model = build_classification_cnn()

    model.summary()

    print("\nMODEL CONFIGURATION")
    print("=" * 45)
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")
    print(f"Total parameters: {model.count_params():,}")
    print(f"Classes: 3")
    print("Class mapping:")
    print("0: glioma")
    print("1: meningioma")
    print("2: pituitary")
