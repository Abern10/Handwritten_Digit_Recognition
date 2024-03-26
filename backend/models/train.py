import tensorflow as tf
import tensorflow_datasets as tfds
from . import model
from sklearn.model_selection import train_test_split

def load_and_preprocess_data():
    # Load EMNIST dataset
    emnist_data, emnist_info = tfds.load('emnist', split='train', as_supervised=True, with_info=True)

    # Preprocess EMNIST data
    def preprocess_image(image, label):
        # Resize image to 28x28 pixels and normalize pixel values
        image = tf.image.resize(image, (28, 28))
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    # Apply preprocessing to each image in the dataset
    emnist_data = emnist_data.map(preprocess_image)

    # Split dataset into training and validation sets
    train_data, val_data = emnist_data.take(60000), emnist_data.skip(60000)

    # Batch and shuffle the training and validation data
    train_data = train_data.shuffle(60000).batch(32)
    val_data = val_data.batch(32)

    return train_data, val_data

def train_model():
    # Load and preprocess data
    train_data, val_data = load_and_preprocess_data()

    # Build the model
    cnn_model = model.build_model()

    # Train the model
    cnn_model.fit(train_data, validation_data=val_data, epochs=10)

    # Save the trained model to a file
    cnn_model.save('emnist_cnn_model.h5')

if __name__ == "__main__":
    train_model()