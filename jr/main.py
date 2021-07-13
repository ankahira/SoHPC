import tensorflow as tf
import pickle
import os
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

if not os.path.exists('data.pkl'):
    from utils import fetch_cifar10
    fetch_cifar10()

elif not os.path.exists('cifar100.pkl'):
    from utils import fetch_cifar100
    fetch_cifar100()

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

bin_file = open('data.pkl', mode='rb')
data = pickle.load(bin_file)
print(len(data))

x_train, y_train, x_test, y_test = data[0], data[1], data[2], data[3]
print(y_test, y_train)

# classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# p = sns.countplot(y_train.numpy())
# p.set(xticklabels=classes)
# p.get_figure().savefig('plot.png')

input_shape = (32, 32, 3)

print(x_train.shape)

#x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 3)
x_train = tf.cast(x_train, tf.float32) / 255.0
#x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 3)
x_test = tf.cast(x_test, tf.float32) / 255.0

y_train = tf.one_hot(y_train, depth=10)
y_test = tf.one_hot(y_test, depth=10)

plt.imsave('sample.png', x_train[100].numpy())

batch_size = 64
num_classes = 10
epochs = 50
#strategy = tf.distribute.MultiWorkerMirroredStrategy()

#with strategy.scope():
model = tf.keras.models.Sequential([
		tf.keras.layers.Conv2D(64, 3, padding='same', input_shape=x_train.shape[1:], activation='relu'),
		tf.keras.layers.BatchNormalization(),
		tf.keras.layers.Conv2D(64, 3, activation='relu'),
		tf.keras.layers.MaxPooling2D(),
		tf.keras.layers.Dropout(0.25),

		tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
		tf.keras.layers.BatchNormalization(),
		tf.keras.layers.Conv2D(128, 3, activation='relu'),
		tf.keras.layers.MaxPooling2D(),
		tf.keras.layers.Dropout(0.25),

		tf.keras.layers.Flatten(),
		tf.keras.layers.Dense(512, activation='relu'),
		tf.keras.layers.Dropout(0.5),
		tf.keras.layers.Dense(num_classes, activation='softmax'),
])

model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-06),
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
