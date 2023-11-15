import streamlit as st

import pickle

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from streamlit_drawable_canvas import st_canvas
# import numpy as np
from PIL import Image
import io
import base64
import torch
from torch import nn
import torch.nn.functional as F
from skorch import NeuralNetClassifier

device = 'cuda' if torch.cuda.is_available() else 'cpu'

st.header("Prediction")

if st.button('Load Model'):
    dynamic_model = torch.load('saved_model.pth', map_location=device)

if st.button("Have you uploaded your drawing on Canvas?"):
    if st.button("Predict the digit drawn"):
        # Load the image
        image_path = "drawing.png"  # Change this to the path of your saved drawing
        image = Image.open(image_path)

        # Resize the image to (28, 28)
        resized_image = image.resize((28, 28))

        # Convert the image to a NumPy array
        image_array = np.array(resized_image)

        # If the image is RGBA (4 channels), convert it to grayscale (1 channel)
        if len(image_array.shape) == 3 and image_array.shape[2] == 4:
            image_array = image_array[:, :, :3]

        # Convert the image to grayscale if it's not already
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            image_array = np.dot(image_array[..., :3], [0.2989, 0.5870, 0.1140])

        # Normalize the pixel values to be in the range [0, 1]
        image_array = image_array / 255.0


