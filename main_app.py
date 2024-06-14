# Library imports
import numpy as np
import streamlit as st
from keras.models import load_model

# Loading the Model
model = load_model('dog_breed.h5')

# Name of Classes
CLASS_NAMES = ['Scottish Deerhound', 'Bernese Mountain Dog', 'Maltese Dog']

# Setting Title of App
st.title("Dog Breed Prediction")
st.markdown("Upload an image of the dog")

# Uploading the dog image
dog_image = st.file_uploader("Choose an image...", type="png")
submit = st.button('Predict')

# On predict button click
if submit:
    if dog_image is not None:
        try:
            # Convert the file to an array
            file_bytes = np.asarray(bytearray(dog_image.read()), dtype=np.uint8)

            # Displaying the image
            st.image(file_bytes, channels="BGR")

            # Make Prediction
            Y_pred = model.predict(file_bytes)
            breed = CLASS_NAMES[np.argmax(Y_pred)]

            st.title(f"The Dog Breed is {breed}")

        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")
            st.write(f"Unexpected error: {str(e)}")

    else:
        st.error("Please upload a valid image file.")
