import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms
from Testing import process
from CellClassifier import CellClassifier
import time
import os
from dotenv import load_dotenv

load_dotenv()
model_path = os.getenv("MODEL_PATH")

if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
    st.session_state.uploader_key = 0
    #st.session_state.reset = False
    st.session_state.classify = False
    st.session_state.classification = None
    st.session_state.already_run = False

#already_run = False


model = CellClassifier()
model.load_state_dict(torch.load(model_path))
model.eval()
device = torch.device("cpu")
model.to(device)

st.title("Leukemia Cell Classifier")

uploaded_file = st.file_uploader("Upload an image of a cell", key = st.session_state.uploader_key)

if uploaded_file is not None:
    st.session_state.uploaded_file = uploaded_file
    st.session_state.classify = True

if st.session_state.uploaded_file is not None and st.session_state.classify:
    #print(st.session_state.reset)
    image = Image.open(st.session_state.uploaded_file)

    st.image(image, caption = 'Uploaded image', use_column_width=True)
    #st.write("Classifying...")
    transformed_image = process(st.session_state.uploaded_file)
    transformed_image = transformed_image.to(device)
    if st.session_state.classify and not st.session_state.already_run:
        placeholder = st.empty()
        print("Classifying")
        placeholder.text("Classifying...")
        time.sleep(2)
        with torch.no_grad():
            output = model(transformed_image)
            _, predicted = torch.max(output, 1)
        st.session_state.classify = False
        if predicted.item() == 0:
            placeholder.markdown("<h2>Cancerous</h2>", unsafe_allow_html=True)
            st.session_state.classification = "Cancerous"
        else:
            placeholder.markdown("<h2>Healthy</h2>", unsafe_allow_html=True)
            st.session_state.classification = "Healthy"
        st.session_state.already_run = True
    st.session_state.uploaded_file = None
    
    if st.button('Reset'): 
        print("A")
        st.session_state.uploaded_file = None
        print("B")
        st.session_state.uploader_key += 1
        print("C")
        st.session_state.already_run = False
        #uploaded_file = None
        #st.session_state.reset = True
        print("Reset clicked:", st.session_state.uploaded_file, st.session_state.uploader_key)
        st.rerun()

