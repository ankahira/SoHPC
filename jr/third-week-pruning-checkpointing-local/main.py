import tensorflow_model_optimization as tfmot
import numpy as np
from time import time

from utils import *
from architecture import create_model

ENABLE_RESTORING_FROM_CHECKPOINT = False
ENABLE_RESTORING_STORED_MODEL = False
BATCH_SIZE = 64
NUM_CLASSES = 10
NUM_EPOCHS = 30

history_data = []
data_for_plots = {
    "test_accuracies": [],
    "test_losses": [],
    "training_times": [],
    "compressed_sizes": []
}

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

data = fetch_cifar10()
x_train, y_train, x_test, y_test = preprocess_data(data)

for model_type in ("baseline", "pruned"):
    model = create_model(x_train.shape[1:])
    callbacks = []
    saved_model_path = f"./{model_type}/model.h5"
    model_restored = False

    if model_type == "pruned":  # configuring pruning and creating pruned model
        end_step = np.ceil(y_train.shape[0] / BATCH_SIZE).astype(np.int32) * NUM_EPOCHS
        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                                     final_sparsity=0.80,
                                                                     begin_step=0,
                                                                     end_step=end_step)
        }
        log_dir = tempfile.mkdtemp()
        model = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)
        callbacks.append(tfmot.sparsity.keras.UpdatePruningStep())
        callbacks.append(tfmot.sparsity.keras.PruningSummaries(log_dir=log_dir))

    if ENABLE_RESTORING_FROM_CHECKPOINT:
        epochs_str = str(NUM_EPOCHS)
        if len(epochs_str) < 2:
            epochs_str = '0' + epochs_str
        restored_checkpoint_path = f"./{model_type}/checkpoint/{epochs_str}"
        if os.path.exists(restored_checkpoint_path + ".index"):
            model.load_weights(restored_checkpoint_path)
            model_restored = True

    if ENABLE_RESTORING_STORED_MODEL:
        if os.path.exists(saved_model_path):
            model = tf.keras.models.load_model(saved_model_path)
            model_restored = True

    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-06),
                  loss='categorical_crossentropy', metrics=['acc'])

    if not model_restored:

        checkpoint_filepath = './' + model_type + '/checkpoint/{epoch:02d}'
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_freq=int(10 * np.ceil(y_train.shape[0] / BATCH_SIZE)),
            save_weights_only=True,
            monitor='accuracy',
            mode='auto')
        callbacks.append(model_checkpoint_callback)
        start = time()
        history = model.fit(x_train, y_train, batch_size=BATCH_SIZE,
                            epochs=NUM_EPOCHS, callbacks=callbacks)
        history_data.append((history, model_type))
        data_for_plots["training_times"].append(time() - start)

        if model_type == "pruned":
            tf.keras.models.save_model(tfmot.sparsity.keras.strip_pruning(model),
                                       saved_model_path, include_optimizer=False)
        else:
            tf.keras.models.save_model(model, saved_model_path, include_optimizer=False)
        data_for_plots["compressed_sizes"].append(get_gzipped_model_size(saved_model_path))

    print(f'\nEvaluation of {model_type} model on test set:')
    test_results = model.evaluate(x_test, y_test)
    data_for_plots["test_losses"].append(test_results[0])
    data_for_plots["test_accuracies"].append(test_results[1])

if not model_restored:
    draw_and_save_training_plots(history_data, "train-loss-acc")
    draw_and_save_bar_charts(data_for_plots)
    #draw_and_save_other_plots(data_for_plots, )


# data - training time, test accuracy, test loss, zipped files size
