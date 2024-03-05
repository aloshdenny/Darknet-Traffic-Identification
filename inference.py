import torch
import numpy as np
from objective_function import obj_fun
from MRFO import OriginalMRFO, IMRFO
from load_save import save, load
from plot_res import plot_res
from classifier import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the training and testing data
X_train = load('X_train')
X_test = load('X_test')
y_train = load('y_train')
y_test = load('y_test')

# Wrap the training data tensors in CUDA tensors
X_train = X_train.to(device)
y_train = y_train.to(device)

# Load the trained model
model = torch.load("models/model_4.0_T1_90_10.pt")

# Make predictions on the test data
predictions = model.predict(X_test)

# Evaluate the model's performance
accuracy = (predictions == y_test).mean()
accuracy()