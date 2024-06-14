# Library imports
import numpy as np
import streamlit as st
from keras.models import load_model
from PIL import Image

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
            # Convert the file to a PIL Image
            pil_image = Image.open(dog_image)
            # Resize the image
            resized_image = pil_image.resize((224, 224))
            # Convert the PIL Image to a NumPy array
            image_array = np.array(resized_image)
            # Normalize the image
            normalized_image = image_array / 255.0
            # Expand dimensions to match the model's input shape
            input_data = np.expand_dims(normalized_image, axis=0)
            
            # Make Prediction
            Y_pred = model.predict(input_data)
            breed = CLASS_NAMES[np.argmax(Y_pred)]

            st.title(f"The Dog Breed is {breed}")

        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")
            st.write(f"Unexpected error: {str(e)}")

    else:
        st.error("Please upload a valid image file.")
