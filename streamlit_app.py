# filename: streamlit_app.py

import streamlit as st
import requests
from PIL import Image
from io import BytesIO

st.title('MedSAM_Lite Model Inference for NPZ Files')

st.write("Upload a .npz file for processing:")

uploaded_file = st.file_uploader("Choose a .npz file...", type=["npz"])
if uploaded_file is not None:
    # Send the .npz file to the FastAPI server
    files = {"file": (uploaded_file.name, uploaded_file, "multipart/form-data")}
    response = requests.post("http://localhost:8000/process_npz/", files=files)

    if response.status_code == 200:
        # Get the byte array of the processed image
        processed_image_bytes = response.content  # Directly use the content of the response

        # Convert the byte array to an image
        processed_image = Image.open(BytesIO(processed_image_bytes))
        st.image(processed_image, caption='Processed Image', use_column_width=True)
    else:
        st.write("Error in processing the .npz file")
