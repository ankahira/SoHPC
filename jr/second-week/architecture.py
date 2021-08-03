import tensorflow as tf


def create_architecture(shape=None, num_classes=1000):
    if shape is None:
        shape = [224, 224, 3]
    print(shape)
    model = tf.keras.models.Sequential([

        tf.keras.layers.Conv2D(64, 3, padding='same', input_shape=shape, activation='relu'),
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(strides=(2, 2), padding='same'),

        tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
        tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
		#tf.keras.layers.MaxPooling2D(strides=(2, 2), padding='same'),

        tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu'),
        tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(strides=(2, 2), padding='same'),

        tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu'),
        tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu'),
        #tf.keras.layers.MaxPooling2D(strides=(2, 2), padding='same'),

        tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu'),
        tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(strides=(2, 2), padding='same'),
        
		tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax'),
    ])
    return model
