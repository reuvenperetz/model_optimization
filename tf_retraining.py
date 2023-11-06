import tensorflow as tf
import keras
import tensorflow as tf
import os
from tensorflow.keras.preprocessing import image_dataset_from_directory
import wandb
from wandb.keras import WandbCallback

wandb.init(project='mct_pruning')

def get_compiled_model(path):
    model = keras.models.load_model(path)
    model.compile(
        optimizer=keras.optimizers.SGD(),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    return model


def get_imagenet_dataset(train_dir, test_dir, img_height, img_width, train_batch_size, val_batch_size):
    # Load the ImageNet dataset from the specified directories
    train_dataset = image_dataset_from_directory(train_dir,
                                                 shuffle=True,
                                                 batch_size=train_batch_size,
                                                 image_size=(img_height, img_width))

    test_dataset = image_dataset_from_directory(test_dir,
                                                shuffle=False,
                                                batch_size=val_batch_size,
                                                image_size=(img_height, img_width))

    # It's common to use a split of the training data for validation.
    # If you have a separate validation set, load it similarly to the training and test sets.

    # Preprocess the data (scale the pixel values to be between 0 and 1)
    # normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)
    #
    # train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
    # test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y))

    return train_dataset, test_dataset  # And also return your validation dataset if you have one.


# Create a MirroredStrategy.
strategy = tf.distribute.MirroredStrategy()
print("Number of devices: {}".format(strategy.num_replicas_in_sync))

path = '/Vols/vol_design/tools/swat/users/reuvenp/ptq/github_repo/my_fork/model_optimization/tmp6y3d2w2s.keras'

# ds = '/data/projects/swat/datasets_src/ImageNet/'
ds = '/local_datasets/ImageNetV2/'
train_dataset, test_dataset = get_imagenet_dataset(ds+'ILSVRC2012_img_train',
                                                   ds+'ILSVRC2012_img_val_TFrecords',
                                                   224, 224,
                                                   512*4, 512*4)


# Prepare a directory to store all the checkpoints.
checkpoint_dir = "./ckpt"
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)


def make_or_restore_model():
    # Either restore the latest model, or create a fresh one
    # if there is no checkpoint available.
    checkpoints = [checkpoint_dir + "/" + name for name in os.listdir(checkpoint_dir)]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print("Restoring from", latest_checkpoint)
        return keras.models.load_model(latest_checkpoint)
    print("Creating a new model")
    return get_compiled_model(path)


def run_training(epochs):
    # Create a MirroredStrategy.
    strategy = tf.distribute.MirroredStrategy()

    # Open a strategy scope and create/restore the model
    with strategy.scope():
        model = make_or_restore_model()

    callbacks = [
        # This callback saves a SavedModel every epoch
        # We include the current epoch in the folder name.
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_dir + "/ckpt-{epoch}", save_freq="epoch"
        ),
        WandbCallback()
    ]



    model.fit(
        train_dataset,
        epochs=1,
        callbacks=callbacks,
        validation_data=test_dataset,
        verbose=2
    )


# Running the first time creates the model
run_training(epochs=120)

