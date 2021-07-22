import tensorflow as tf
import tensorflow_datasets as tfds
from matplotlib import pyplot as plt
import pickle
import os
import zipfile
import tempfile


def fetch_mnist():
    print('fetching data...')
    (x_train, y_train), (x_test, y_test) = tfds.load('mnist', split=['train', 'test'],
                                                     as_supervised=True,
                                                     batch_size=-1)
    data = [x_train, y_train, x_test, y_test]
    bin_file = open('data.pkl', mode='wb')
    pickle.dump(data, bin_file)
    return data


def fetch_cifar10():
    print('fetching data...')
    (x_train, y_train), (x_test, y_test) = tfds.load('cifar10', split=['train', 'test'],
                                                     as_supervised=True,
                                                     batch_size=-1)
    data = [x_train, y_train, x_test, y_test]
    bin_file = open('data.pkl', mode='wb')
    pickle.dump(data, bin_file)
    return data


def preprocess_data(data):
    return tf.cast(data[0], tf.float32) / 255.0, tf.one_hot(data[1], depth=10), \
           tf.cast(data[2], tf.float32) / 255.0, tf.one_hot(data[3], depth=10)


def draw_and_save_training_plots(history_tuples, filename):
    fig, ax = plt.subplots(2, 1)
    for history, name in history_tuples:
        ax[0].plot(range(1, len(history.history['loss']) + 1), history.history['loss'], label=f"{name} training Loss")
        ax[1].plot(range(1, len(history.history['loss']) + 1), history.history['acc'], label=f"{name} training Accuracy")

    ax[0].legend(loc='best', shadow=True)
    ax[0].set_xlabel("Epoch number")
    ax[0].set_ylabel("Cross entropy loss")

    ax[1].legend(loc='best', shadow=True)
    ax[1].set_xlabel("Epoch number")
    ax[1].set_ylabel("Accuracy on training set")

    fig.tight_layout()
    plt.savefig(f'{filename}.png')


def draw_and_save_bar_charts(data, filename="bar-charts"):
    labels = ["baseline", "pruned"]
    fig, ax = plt.subplots(2, 2)

    ax[0, 0].bar(range(2), data["test_accuracies"], tick_label=labels, color=["red", "blue"])
    ax[0, 0].set_xlabel("Model")
    ax[0, 0].set_ylabel("Accuracy")
    ax[0, 0].set_title("Accuracy on test set")

    ax[0, 1].bar(range(2), data["test_losses"], tick_label=labels, color=["red", "blue"])
    ax[0, 1].set_xlabel("Model")
    ax[0, 1].set_ylabel("Loss")
    ax[0, 1].set_title("Loss on test set")

    ax[1, 0].bar(range(2), data["training_times"], tick_label=labels, color=["red", "blue"])
    ax[1, 0].set_xlabel("Model")
    ax[1, 0].set_ylabel("Time [s]")
    ax[1, 0].set_title("Training time")

    ax[1, 1].bar(range(2), data["compressed_sizes"], tick_label=labels, color=["red", "blue"])
    ax[1, 1].set_xlabel("Model")
    ax[1, 1].set_ylabel("Size [MB]")
    ax[1, 1].set_title("Zipped model size")

    fig.tight_layout()
    plt.savefig(f'{filename}.png')


def draw_and_save_other_plots(data, x, filename="plots"):
    fig, ax = plt.subplots(2, 2)

    ax[0, 0].plot(x, data["test_accuracies"])
    ax[0, 0].set_xlabel("Final model sparsity")
    ax[0, 0].set_ylabel("Accuracy")
    ax[0, 0].set_title("Accuracy on test set")

    ax[0, 1].plot(x, data["test_losses"])
    ax[0, 1].set_xlabel("Final model sparsity")
    ax[0, 1].set_ylabel("Loss")
    ax[0, 1].set_title("Loss on test set")

    ax[1, 0].plot(x, data["training_times"])
    ax[1, 0].set_xlabel("Final model sparsity")
    ax[1, 0].set_ylabel("Time [s]")
    ax[1, 0].set_title("Training time")

    ax[1, 1].plot(x, data["compressed_sizes"])
    ax[1, 1].set_xlabel("Final model sparsity")
    ax[1, 1].set_ylabel("Size [MB]")
    ax[1, 1].set_title("Zipped model size")

    fig.tight_layout()
    plt.savefig(f'{filename}.png')


def get_gzipped_model_size(file):
    _, zipped_file = tempfile.mkstemp('.zip')
    with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
        f.write(file)

    return os.path.getsize(zipped_file) / 1024 ** 2
