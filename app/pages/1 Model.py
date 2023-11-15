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


def save_drawing(image_data):
    # Convert the base64 image data to a NumPy array
    image_array = decode_base64(image_data)

    # Resize the image to (28, 28)
    resized_image = resize_image(image_array, target_shape=(28, 28))

    # Save the resized image
    image = Image.fromarray(resized_image)
    image.save("saved_drawing.png")
    st.success("Drawing saved as saved_drawing.png")

def decode_base64(base64_string):
    image_data = base64_string.split(",")[1]
    image_bytes = io.BytesIO(base64.b64decode(image_data))
    image_array = np.array(Image.open(image_bytes))
    return image_array

def resize_image(image_array, target_shape):
    # Resize the image using PIL
    image = Image.fromarray(image_array)
    resized_image = image.resize(target_shape[::-1], Image.ANTIALIAS)
    return np.array(resized_image)

st.title('Hand written text classification ')

st.subheader("Data Preprocessing")

# print("Loading the Data...")

load_data_text=st.text("Loading the Data...")
# mnist = fetch_openml('mnist_784', as_frame=False, cache=False)
# Load the dataset from the binary file
with open('/home/kanishk/Documents/mnist_data.pkl', 'rb') as file:
    mnist = pickle.load(file)
load_data_text.empty()
load_data_text=st.text("Data Loaded")

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

split_ratio = st.slider('Test Size', min_value=0.01, max_value=0.99, step=0.01)  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-split_ratio, random_state=42)

XCnn = X.reshape(-1, 1, 28, 28)
# st.text("Shape of Training Data: ", XCnn.shape)
XCnn_train, XCnn_test, y_train, y_test = train_test_split(XCnn, y, test_size=0.25, random_state=42)
# XCnn_train.shape, y_train.shape

st.write('Training set shape:',XCnn_train.shape, y_train.shape)
st.write('Training set shape:',XCnn_test.shape, y_test.shape)

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

st.subheader("Model Training")

# num_layers = st.number_input('Enter the number of layers:', step=1)

#     # Get the number of neurons in each layer from the user
# num_neurons = [st.number_input(f'Number of neurons in Layer {i+1}', step=1) for i in range(num_layers)]

# User input for number of layers
num_layers = st.slider("Number of Layers", min_value=1, max_value=10, value=3, key='num_layers')

# Collect information for each layer
layers_info = []
j=0
for i in range(num_layers):
    st.header(f"Layer {i + 1}")
    layer_type = st.selectbox(f"Layer Type {i + 1}", ["conv", "fc"], key=f'layer_type_{i}')
    if layer_type=="conv":

        if i==0:
            in_channels = st.number_input(f"Input Channels {i + 1}", min_value=1, value=1, key=f'in_channels_{i}')
            in_features = 28
        else :
            in_channels = layers_info[i-1]['out_channels']
            in_features=layers_info[i-1]['out_features']
        out_channels = st.number_input(f"Output Channels {i + 1}", min_value=1, value=32, key=f'out_channels_{i}')
        kernel_size = st.number_input(f"Kernel Size {i + 1}", min_value=1, value=3, key=f'kernel_size_{i}')
        # in_features=layers_info[i-1]['out_features']
        

        # st.number_input(f"Input Features {i + 1}", min_value=1, value=1600, key=f'in_features_{i}')
        out_features = ((in_features-kernel_size)+1)//2
        # st.number_input(f"Output Features {i + 1}", min_value=1, value=100, key=f'out_features_{i}')
    if layer_type=="fc":
        in_channels = 1
        out_channels = 1
        kernel_size = 1
        if j==0:
            if i==0:
                in_features=28*28*1
            else:
                in_features=(layers_info[i-1]['out_features']**2)*layers_info[i-1]['out_channels']
        else:
            in_features = (layers_info[i-1]['out_features'])

        # st.number_input(f"Input Features {i + 1}", min_value=1, value=1600, key=f'in_features_{i}')
        out_features = st.number_input(f"Nodes {i + 1}", min_value=1, value=100, key=f'out_features_{i}')
        j=j+1

    layer_info = {
        'type': layer_type,
        'in_channels': in_channels,
        'out_channels': out_channels,
        'kernel_size': kernel_size,
        'in_features': in_features,
        'out_features': out_features
    }
    st.write(in_features)

    layers_info.append(layer_info)
    

# User input for dropout
dropout = st.slider("Dropout", min_value=0.0, max_value=1.0, value=0.5, key='dropout')


import torch
from torch import nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# mnist_dim = X.shape[1]
# hidden_dim = int(mnist_dim/8)
# output_dim = len(np.unique(mnist.target))

class Cnn(nn.Module):
    def __init__(self, layers_info, dropout=0.5):
        super(Cnn, self).__init__()

        self.conv_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()

        for i, layer_info in enumerate(layers_info):
            if layer_info['type'] == 'conv':
                in_channels = layers_info[i]['in_channels']
                # layer_info.get('in_channels') if i == 0 else layers_info[i - 1]['out_channels']
                layer = nn.Conv2d(int(in_channels), int(layer_info['out_channels']), kernel_size=int(layer_info['kernel_size']))
                self.conv_layers.append(layer)
            elif layer_info['type'] == 'fc':
                in_features = layer_info.get('in_features') 
                # if i == 0 else layers_info[i - 1]['out_features']
                layer = nn.Linear(int(in_features), int(layer_info['out_features']))
                self.fc_layers.append(layer)
            else:
                raise ValueError("Invalid layer type. Supported types: 'conv' or 'fc'.")

            # self.layers.append(layer)
        
        self.conv2_drop = nn.Dropout2d(p=dropout)
        
        self.fc_last = nn.Linear(int(layers_info[-1]['out_features']), 10)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        for layer in self.conv_layers:
            x = torch.relu(F.max_pool2d(layer(x), 2))
            # st.write(x.shape)
        # x = torch.relu(F.max_pool2d(self.conv2_drop(x), 2))
        # st.write(x.shape)
        x = x.view(-1, x.size(1) * x.size(2) * x.size(3))
        # st.write(x.shape)
        for layer in self.fc_layers:
            x = torch.relu(layer(x))
            # st.write(x.shape)
        # flatten over channel, height, and width
        # x = x.view(-1, x.size(1) * x.size(2) * x.size(3))

        x = torch.relu(self.dropout(x))
        x = torch.softmax(self.fc_last(x), dim=-1)
        return x

# class ClassifierModule(nn.Module):
#     def __init__(
#             self,
#             input_dim=mnist_dim,
#             hidden_dim=hidden_dim,
#             output_dim=output_dim,
#             dropout=0.5,
#     ):
#         super(ClassifierModule, self).__init__()
#         self.dropout = nn.Dropout(dropout)

#         self.hidden = nn.Linear(input_dim, hidden_dim)
#         self.output = nn.Linear(hidden_dim, output_dim)

#     def forward(self, X, **kwargs):
#         X = F.relu(self.hidden(X))
#         X = self.dropout(X)
#         X = F.softmax(self.output(X), dim=-1)
#         return X
    
# class DynamicClassifierModule(nn.Module):
#     def __init__(self, input_dim, num_layers, num_neurons, output_dim, dropout=0.5):
#         super(DynamicClassifierModule, self).__init__()
#         self.dropout = nn.Dropout(dropout)

#         # Create layers dynamically based on user input
#         layers = [nn.Linear(input_dim, num_neurons[0])]
#         layers.append(nn.ReLU())
#         layers.append(self.dropout)

#         for i in range(num_layers - 1):
#             layers.append(nn.Linear(num_neurons[i], num_neurons[i + 1]))
#             layers.append(nn.ReLU())
#             layers.append(self.dropout)

#         layers.pop()  # Remove the last ReLU and dropout
#         self.hidden = nn.Sequential(*layers)
#         self.output = nn.Linear(num_neurons[-1], output_dim)

#     def forward(self, X, **kwargs):
#         X = self.hidden(X)
#         X = self.dropout(X)
#         X = F.softmax(self.output(X), dim=-1)
#         return X


from skorch import NeuralNetClassifier
torch.manual_seed(0)



# # net = NeuralNetClassifier(
# #     DynamicClassifierModule,
# #     max_epochs=1,
# #     lr=0.1,
# #     device=device,
# # )

max_epochs = st.number_input('Enter the number of epochs', step=1)
lr = st.number_input('Enter the learning rate')

# model = Cnn(layers_info, dropout)

# Instantiate your model
cnn_model = Cnn(layers_info, dropout)

# Instantiate NeuralNetClassifier with your model
dynamic_model = NeuralNetClassifier(module=cnn_model,
                                    max_epochs=1,
                                    lr=lr,
                                    device=device)

from sklearn.metrics import accuracy_score

bt=st.slider('Train Model',0,1,0,1)

if bt:
    progress_bar = st.progress(0)

    # Lists to store training and validation loss for plotting
    train_losses = []
    valid_losses = []

    progress_text=st.text("Progress:\n")

    st.text("Epoch    Train Loss    Valid Acc    Valid Loss    Duration")
    param_text=st.text("-------  ------------  -----------  ------------  ------")

    for epoch in range(max_epochs):
        # param_text.empty()
        # You might want to update the training data based on your specific use case
        dynamic_model.fit(XCnn_train, y_train)  # <-- Add this line to fit the model
        
        # Get training and validation metrics
        train_loss = dynamic_model.history[-1, 'train_loss']
        # train_loss = dynamic_model.history[-1, 'train_acc']
        valid_acc = dynamic_model.history[-1, 'valid_acc']
        valid_loss = dynamic_model.history[-1, 'valid_loss']
        dur = dynamic_model.history[-1, 'dur']

        # Print the metrics
        st.text(f"{epoch + 1:6d}        {train_loss:.4f}        {valid_acc:.4f}        {valid_loss:.4f}         {dur:.4f}")

        # Append training and validation loss for plotting
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        progress_bar.progress((epoch + 1) / max_epochs)  # Update the progress bar
        # Line chart for training and validation loss
    train_chart=st.line_chart({"Training Loss": train_losses, "Validation Loss": valid_losses})
    st.success('Model training completed!')
    y_pred_cnn=dynamic_model.predict(XCnn_test)
    st.write(accuracy_score(y_test, y_pred_cnn))
    # if st.button("Save Model"):
    #     # Save the model
    #     torch.save(dynamic_model.module_, 'saved_model.pth')
    #     st.success("Model saved as saved_model.pth")

    # # progress_text.empty()
if st.slider("Predict the digit drawn",0,1,0,1):
    # Load the image
    image_path = "drawing.png"  # Change this to the path of your saved drawing
    image = Image.open(image_path)
    # st.image(image)
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
    st.image(image_array)

    image_array = image_array.reshape(1, 1, 28, 28)
    image_array = image_array.astype(XCnn_test.dtype)
    # st.write(type(XCnn_test))
    # st.write(type(image_array))
    XCnn_test = np.vstack([XCnn_test, image_array])
    # XCnn_test.shape
    y_pred_cnn=dynamic_model.predict(XCnn_test)
    st.write(y_pred_cnn[-1])


