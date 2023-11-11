import streamlit as st

import pickle

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


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

st.subheader("Model Training")

num_layers = st.number_input('Enter the number of layers:', step=1)

    # Get the number of neurons in each layer from the user
num_neurons = [st.number_input(f'Number of neurons in Layer {i+1}', step=1) for i in range(num_layers)]

import torch
from torch import nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

mnist_dim = X.shape[1]
hidden_dim = int(mnist_dim/8)
output_dim = len(np.unique(mnist.target))

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
    
class DynamicClassifierModule(nn.Module):
    def __init__(self, input_dim, num_layers, num_neurons, output_dim, dropout=0.5):
        super(DynamicClassifierModule, self).__init__()
        self.dropout = nn.Dropout(dropout)

        # Create layers dynamically based on user input
        layers = [nn.Linear(input_dim, num_neurons[0])]
        layers.append(nn.ReLU())
        layers.append(self.dropout)

        for i in range(num_layers - 1):
            layers.append(nn.Linear(num_neurons[i], num_neurons[i + 1]))
            layers.append(nn.ReLU())
            layers.append(self.dropout)

        layers.pop()  # Remove the last ReLU and dropout
        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(num_neurons[-1], output_dim)

    def forward(self, X, **kwargs):
        X = self.hidden(X)
        X = self.dropout(X)
        X = F.softmax(self.output(X), dim=-1)
        return X


from skorch import NeuralNetClassifier

torch.manual_seed(0)



# net = NeuralNetClassifier(
#     DynamicClassifierModule,
#     max_epochs=1,
#     lr=0.1,
#     device=device,
# )

# net.fit(X_train, y_train);

# def train_model(net, X_train, y_train, progress_bar):
#     max_epochs=net.max_epochs
#     for epoch in range(1, max_epochs + 1):
#         net.partial_fit(X_train, y_train)
#         progress_bar.progress(epoch / max_epochs)
#         time.sleep(0.1)

# if st.button('Fit Model'):
#     my_bar = st.progress(0)
#     # max_epochs = 20

#     with st.spinner('Training model...'):
#         train_model(net, X_train, y_train, my_bar)

#     st.success('Training complete!')

max_epochs = st.number_input('Enter the number of epochs', step=1)
lr = st.number_input('Enter the learning rate')

dynamic_model = NeuralNetClassifier(DynamicClassifierModule(mnist_dim,num_layers,num_neurons,output_dim),
                                        max_epochs=1,
                                        lr=lr,
                                        device=device,
                                        )

 # Train the model for a fixed number of epochs (e.g., 5)
# if st.button('Train Model'):
#     progress_bar = st.progress(0)
#     for epoch in range(max_epochs):
#         # progress_text.empty()
#         # You might want to update the training data based on your specific use case
#         dynamic_model.fit(X_train, y_train)

#         progress_bar.progress((epoch + 1) / max_epochs)  # Update the progress bar
#     st.success('Model training completed!')

from sklearn.metrics import accuracy_score

if st.button('Train Model'):
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
        dynamic_model.fit(X_train, y_train)  # <-- Add this line to fit the model
        
        # Get training and validation metrics
        train_loss = dynamic_model.history[-1, 'train_loss']
        valid_acc = dynamic_model.history[-1, 'valid_acc']
        valid_loss = dynamic_model.history[-1, 'valid_loss']
        dur = dynamic_model.history[-1, 'dur']

        # Print the metrics
        st.text(f"{epoch + 1:6d}   {train_loss:.4f}         {valid_acc:.4f}        {valid_loss:.4f}         {dur:.4f}")

        # Append training and validation loss for plotting
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        progress_bar.progress((epoch + 1) / max_epochs)  # Update the progress bar
        # Line chart for training and validation loss
    train_chart=st.line_chart({"Training Loss": train_losses, "Validation Loss": valid_losses})
    st.success('Model training completed!')
    progress_text.empty()

if st.button('Save Model'):
    torch.save(dynamic_model, 'saved_model.pth')

if st.button('Load Model'):
    dynamic_model = torch.load('saved_model.pth')

# # Now you can make predictions after the model has been trained
# y_pred = dynamic_model.predict(X_test)


y_pred = dynamic_model.predict(X_test)

test_acc=accuracy_score(y_test, y_pred)

st.write(f"Test Accuracy: {test_acc*100:.4f}%")

error_mask = y_pred != y_test

plot_example(X_test[error_mask], y_pred[error_mask])




