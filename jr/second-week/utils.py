import tensorflow_datasets as tfds
import pickle


def fetch_cifar10():
    print('fetching data...')
    (x_train, y_train), (x_test, y_test) = tfds.load('cifar10', split=['train', 'test'],
                                                     as_supervised=True,
                                                     batch_size=-1)
    data = [x_train, y_train, x_test, y_test]
    bin_file = open('data.pkl', mode='wb')
    pickle.dump(data, bin_file)


def fetch_cifar100():
    print('fetching data...')
    (x_train, y_train), (x_test, y_test) = tfds.load('cifar100', split=['train', 'test'],
                                                     as_supervised=True,
                                                     batch_size=-1)
    data = [x_train, y_train, x_test, y_test]
    bin_file = open('cifar100.pkl', mode='wb')
    pickle.dump(data, bin_file)
