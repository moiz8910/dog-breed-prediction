# #Library imports
# import numpy as np
# import streamlit as st
# import cv2
# from keras.models import load_model


# #Loading the Model
# model = load_model('dog_breed.h5')

# #Name of Classes
# CLASS_NAMES = ['Scottish Deerhound','Bernese Mountain Dog','Maltese Dog']

# #Setting Title of App
# st.title("Dog Breed Prediction")
# st.markdown("Upload an image of the dog")

# #Uploading the dog image
# dog_image = st.file_uploader("Choose an image...", type="png")
# submit = st.button('Predict')
# #On predict button click
# if submit:


#     if dog_image is not None:

#         # Convert the file to an opencv image.
#         file_bytes = np.asarray(bytearray(dog_image.read()), dtype=np.uint8)
#         opencv_image = cv2.imdecode(file_bytes, 1)



#         # Displaying the image
#         st.image(opencv_image, channels="BGR")
#         #Resizing the image
#         opencv_image = cv2.resize(opencv_image, (224,224))
#         #Convert image to 4 Dimension
#         opencv_image.shape = (1,224,224,3)
#         #Make Prediction
#         Y_pred = model.predict(opencv_image)

#         st.title(str("The Dog Breed is "+CLASS_NAMES[np.argmax(Y_pred)]))

#     else:
#         print ("The photo is not correct")


# Library imports
import numpy as np
import streamlit as st
import cv2
from keras.models import load_model
import sys

# Configure buffering for stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

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
            # Convert the file to an opencv image
            file_bytes = np.asarray(bytearray(dog_image.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)

            # Displaying the image
            st.image(opencv_image, channels="BGR")
            
            # Resizing the image
            opencv_image = cv2.resize(opencv_image, (224, 224))
            
            # Convert image to 4 Dimensions
            opencv_image = np.expand_dims(opencv_image, axis=0)
            
            # Make Prediction
            Y_pred = model.predict(opencv_image)
            breed = CLASS_NAMES[np.argmax(Y_pred)]
            
            st.title(f"The Dog Breed is {breed}")
        
        except BrokenPipeError:
            st.error("An error occurred while making the prediction. Please try again.")
            sys.stderr.write("BrokenPipeError encountered during prediction.\n")
            sys.stderr.flush()
        
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")
            sys.stderr.write(f"Unexpected error: {str(e)}\n")
            sys.stderr.flush()

    else:
        st.error("Please upload a valid image file.")
