import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dropout, Activation, Conv2D, GlobalAveragePooling2D
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
from PIL import Image

# Load the CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Normalize the data
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# One-hot encode the labels
Y_train = to_categorical(y_train)
Y_test = to_categorical(y_test)

# Model function
def allcnn(weights=None):
    model = Sequential()
    model.add(Input(shape=(32, 32, 3)))
    model.add(Conv2D(96, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(96, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(96, (3, 3), padding='same', strides=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(192, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(192, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(192, (3, 3), padding='same', strides=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(192, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(192, (1, 1), padding='valid'))
    model.add(Activation('relu'))
    model.add(Conv2D(10, (1, 1), padding='valid'))
    model.add(GlobalAveragePooling2D())
    model.add(Activation('softmax'))

    if weights:
        model.load_weights(weights)

    return model

# Load pre-trained weights
weights = 'all_cnn_weights_0.9088_0.4994.hdf5'  # Change this if needed
model = allcnn(weights)

# Define optimizer and compile model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'])

# Define class labels
names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
class_labels = dict(zip(range(0, 10), names))

# Function to process the image, make predictions, and display them
def process_and_predict_image(uploaded_file):
    image = Image.open(uploaded_file)
    image = image.resize((32, 32))  # Resize image to match CIFAR-10 input size
    image = np.array(image)  # Convert image to numpy array
    image = image.astype('float32') / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    prediction = model.predict(image)
    predicted_class = np.argmax(prediction, axis=-1)[0]  # Get predicted class index
    predicted_label = class_labels[predicted_class]  # Get the label for the predicted class

    return image[0], predicted_label

# Streamlit interface
st.title("CIFAR-10 Image Classification")

st.write("This app uses a CNN model trained on the CIFAR-10 dataset to predict image classes.")

# File uploader for user to upload an image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
    
    # Button to make predictions
    if st.button("Make Prediction"):
        img, label = process_and_predict_image(uploaded_file)
        st.image(img, caption=f"Predicted Label: {label}", use_column_width=True)
        st.write(f"Prediction: {label}")
