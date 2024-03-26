from app import app
from flask import render_template, request
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the trained EMNIST model
model = tf.keras.models.load_model('../backend/models/emnist_cnn_model.h5')

# Define route for handling predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get uploaded image file from request
    file = request.files['file']

    # Preprocess the image
    img = Image.open(file)
    img = img.convert('L')  # Convert image to grayscale
    img = img.resize((28, 28))  # Resize image to 28x28 pixels
    img_array = np.array(img)  # Convert image to numpy array
    img_array = img_array / 255.0  # Normalize pixel values to range [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction using the EMNIST model
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    # Print prediction result to terminal
    print("Predicted class:", predicted_class)

    return render_template('index.html', prediction=predicted_class)

@app.route('/')
def index():
    return render_template('index.html')