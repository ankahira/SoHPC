import os
import pickle
import tensorflow as tf
from matplotlib import pyplot as plt

from architecture import create_architecture

# import PIL.Image
# import numpy as np
# from matplotlib import image

# img = np.asarray(PIL.Image.open("img.jpg"))
# model = create_architecture(img.shape)
# img = np.expand_dims(img, 0)
# x_train = tf.convert_to_tensor(img)

if not os.path.exists('data.pkl'):
    from utils import fetch_cifar10
    fetch_cifar10()

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

bin_file = open('data.pkl', mode='rb')
data = pickle.load(bin_file)
print(len(data))

x_train, y_train, x_test, y_test = data[0], data[1], data[2], data[3]
print(y_test, y_train)

input_shape = (32, 32, 3)

x_train = tf.cast(x_train, tf.float32) / 255.0
x_test = tf.cast(x_test, tf.float32) / 255.0

y_train = tf.one_hot(y_train, depth=10)
y_test = tf.one_hot(y_test, depth=10)

plt.imsave('sample.png', x_train[100].numpy())

batch_size = 64
num_classes = 10
epochs = 50

model = create_architecture(x_train.shape[1:], 10)
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.000005, decay=1e-06),
              loss='categorical_crossentropy', metrics=['acc'])

history = model.fit(x_train, y_train, batch_size=batch_size,
                    epochs=epochs)

fig, ax = plt.subplots(2, 1)
ax[0].plot(history.history['loss'], color='b', label="Training Loss")
legend = ax[0].legend(loc='best', shadow=True)
ax[0].set_xlabel("Epoch number")
ax[0].set_ylabel("Cross entropy loss")

ax[1].plot(history.history['acc'], color='b', label="Training Accuracy")
legend = ax[1].legend(loc='best', shadow=True)
ax[1].set_xlabel("Epoch number")
ax[1].set_ylabel("Accuracy on training set")

fig.tight_layout()

plt.savefig('loss-acc.png')
print('\nEvaluation on test set:')
test_loss, test_acc = model.evaluate(x_test, y_test)

# labels = [[0] * 1000]
# labels[0][995] = 1
#
# y_train = np.array(labels)
#
# history = model.fit(x_train, y_train, batch_size=1, epochs=10)
