
import tensorflow as tf
from tensorflow.keras import layers, models


@tf.keras.utils.register_keras_serializable(package="BrainTumor")
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])

    intersection = tf.reduce_sum(y_true * y_pred)

    return (
        2.0 * intersection + smooth
    ) / (
        tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth
    )


@tf.keras.utils.register_keras_serializable(package="BrainTumor")
def dice_loss(y_true, y_pred):
    return 1.0 - dice_coefficient(y_true, y_pred)


@tf.keras.utils.register_keras_serializable(package="BrainTumor")
def combined_segmentation_loss(y_true, y_pred):
    binary_loss = tf.keras.losses.binary_crossentropy(
        y_true,
        y_pred
    )

    return tf.reduce_mean(binary_loss) + dice_loss(
        y_true,
        y_pred
    )


def convolution_block(inputs, filters, dropout_rate=0.0):
    x = layers.Conv2D(
        filters,
        kernel_size=3,
        padding="same",
        kernel_initializer="he_normal"
    )(inputs)

    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(
        filters,
        kernel_size=3,
        padding="same",
        kernel_initializer="he_normal"
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate)(x)

    return x


def build_unet(input_shape=(256, 256, 1)):
    inputs = layers.Input(shape=input_shape)

    encoder1 = convolution_block(inputs, 32)
    pool1 = layers.MaxPooling2D(pool_size=2)(encoder1)

    encoder2 = convolution_block(pool1, 64)
    pool2 = layers.MaxPooling2D(pool_size=2)(encoder2)

    encoder3 = convolution_block(pool2, 128)
    pool3 = layers.MaxPooling2D(pool_size=2)(encoder3)

    encoder4 = convolution_block(pool3, 256, 0.1)
    pool4 = layers.MaxPooling2D(pool_size=2)(encoder4)

    bottleneck = convolution_block(pool4, 512, 0.2)

    decoder4 = layers.Conv2DTranspose(
        256,
        kernel_size=2,
        strides=2,
        padding="same"
    )(bottleneck)

    decoder4 = layers.Concatenate()([decoder4, encoder4])
    decoder4 = convolution_block(decoder4, 256, 0.1)

    decoder3 = layers.Conv2DTranspose(
        128,
        kernel_size=2,
        strides=2,
        padding="same"
    )(decoder4)

    decoder3 = layers.Concatenate()([decoder3, encoder3])
    decoder3 = convolution_block(decoder3, 128)

    decoder2 = layers.Conv2DTranspose(
        64,
        kernel_size=2,
        strides=2,
        padding="same"
    )(decoder3)

    decoder2 = layers.Concatenate()([decoder2, encoder2])
    decoder2 = convolution_block(decoder2, 64)

    decoder1 = layers.Conv2DTranspose(
        32,
        kernel_size=2,
        strides=2,
        padding="same"
    )(decoder2)

    decoder1 = layers.Concatenate()([decoder1, encoder1])
    decoder1 = convolution_block(decoder1, 32)

    outputs = layers.Conv2D(
        1,
        kernel_size=1,
        activation="sigmoid",
        name="segmentation_mask"
    )(decoder1)

    model = models.Model(
        inputs=inputs,
        outputs=outputs,
        name="Brain_Tumor_UNet"
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-4
        ),
        loss=combined_segmentation_loss,
        metrics=[
            dice_coefficient,
            tf.keras.metrics.BinaryAccuracy(
                name="binary_accuracy"
            )
        ]
    )

    return model


if __name__ == "__main__":
    model = build_unet()

    model.summary()

    print("\nU-NET MODEL CONFIGURATION")
    print("=" * 55)
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")
    print(f"Total parameters: {model.count_params():,}")
    print("Output activation: Sigmoid")
    print("Segmentation type: Binary")
    print("Model created successfully.")
