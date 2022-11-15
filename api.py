import numpy as np
import streamlit as st
import cv2
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras.utils import load_img, img_to_array
from PIL import Image




st.title("Brain Tumour Predictor 🧠")
model = load_model('brain_tumor_detector.h5')
img = st.file_uploader("Upload an image")

if img:
    st.image(img)
    image = load_img(img, target_size=(224,224))
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array,axis=0)

if st.button("Predict"):
    y_hat = model.predict(img_array)
    y_hat_abs = y_hat.round()
    condlist=[y_hat_abs==1,y_hat_abs==0]
    choicelist=["Is Tumor","Not a Tumor"]
    pred = np.select(condlist,choicelist, default="Not a valid image")

    st.title(f"Prediction: {pred.item()}\nConfidence: {y_hat.max()}")