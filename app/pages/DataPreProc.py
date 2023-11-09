import streamlit as st

import pickle

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.title('Hand written text classification ')

st.subheader("Data Preprocessing")

# print("Loading the Data...")

# st.write("Loading the Data...")

# Load the dataset from the binary file
with open('mnist_data.pkl', 'rb') as file:
    mnist = pickle.load(file)

st.write("Data Loaded")

# print(mnist.data.shape)

st.write("Shape of data: ",mnist.data.shape)

st.write("Select the sample size")

n = st.slider('Number of samples', min_value=2, max_value=70000)

# Get the indices of randomly selected samples
random_indices = np.random.choice(len(mnist.data), size=n, replace=False)

# Select the data and target labels based on the random indices
selected_data = mnist.data[random_indices]
selected_target = mnist.target[random_indices]

st.write("Shape of the selected data: ",selected_data.shape)


X = selected_data.astype('float32')
y = selected_target.astype('int64')

X /= 255.0

# X.min(), X.max()

st.write('Select the train-test split ratio')

split_ratio = st.slider('Test Size', min_value=0.01, max_value=0.99, step=0.05)  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-split_ratio, random_state=42)

st.write('Training set shape:',X_train.shape, y_train.shape)
st.write('Training set shape:',X_test.shape, y_test.shape)

def plot_example(X, y):
    """Plot the first 100 images in a 10x10 grid."""
    plt.figure(figsize=(15, 15))  # Set figure size to be larger (you can adjust as needed)

    for i in range(10):  # For 10 rows
        for j in range(10):  # For 10 columns
            index = i * 10 + j
            plt.subplot(10, 10, index + 1)  # 10 rows, 10 columns, current index
            plt.imshow(X[index].reshape(28, 28))  # Display the image
            plt.xticks([])  # Remove x-ticks
            plt.yticks([])  # Remove y-ticks
            plt.title(y[index], fontsize=8)  # Display the label as title with reduced font size

    plt.subplots_adjust(wspace=0.5, hspace=0.5)  # Adjust spacing (you can modify as needed)
    plt.tight_layout()  # Adjust the spacing between plots for better visualization
    plt.show()  # Display the entire grid
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()


if st.button('Show Training Data'): 
    plot_example(X_train, y_train)


