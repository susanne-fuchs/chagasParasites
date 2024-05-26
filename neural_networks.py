import os.path

import tensorflow as tf
from tensorflow.keras import datasets, layers, models

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def create_datasets_and_train_model(data_dir="Data"):
    """
    Creates a train and validation dataset from a directory structure
    containing the directories 'positive' and 'negative', and uses them to
    train a convolutional neural network.
    :param data_dir: Path containing 'positive' and 'negative' subdirectories.
    """
    batch_size = 3
    positives_path = os.path.join(data_dir,  "positive")
    image_file = os.path.join(positives_path,  os.listdir(positives_path)[0])
    img = mpimg.imread(image_file)
    img_height = img.shape[0]
    img_width = img.shape[1]

    train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        label_mode="binary",
        validation_split=0.2,
        subset="both",
        seed=42,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    class_names = train_ds.class_names
    plot_example_images(train_ds, class_names)

    # Normalize datasets:
    normalization_layer = layers.Rescaling(1. / 255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    model = create_model()

    model.compile(
        optimizer='adam', loss='binary_crossentropy',  metrics=['accuracy'])

    epochs = 10
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

    plot_training_history(history, epochs)

    print(model.summary())

    model_path = "Models"
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model.save(os.path.join(model_path, "cnn_model.keras"))



def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model


def plot_training_history(history, epochs):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig("Results/model_performance.png")


def plot_example_images(dataset, class_names):
    plt.figure(figsize=(10, 5))
    for images, labels in dataset.take(1):
        for i in range(2):
            ax = plt.subplot(1, 2, i+1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[i])
            plt.axis("off")
    plt.tight_layout()
    plt.savefig("Results/example_image.png")
