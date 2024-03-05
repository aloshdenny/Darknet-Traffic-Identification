import numpy as np
from load_save import *
from classifier import *
import matplotlib.pyplot as plt

def obj_fun(soln):
    X_train=load('cur_X_train')
    X_test=load('cur_X_test')
    y_train=load('cur_y_train')
    y_test=load('cur_y_test')

    # Feature selection
    soln = np.round(soln)
    X_train=X_train[:,np.where(soln==1)[0]]
    X_test = X_test[:, np.where(soln == 1)[0]]
    pred, met = prop_classifier(X_train, y_train, X_test, y_test)
    fit = 1/met[0]

    return fit
'''
import pickle

# Load the binary solution vector from the .pkl file
with open('DeepthyJ/Saved data/feat_sel_best_pos.pkl', 'rb') as file:
    soln = pickle.load(file)

# Define or load your feature selection solution
# You should replace this line with your code to obtain the solution vector
solution_vector = obj_fun(soln)  # Replace with your actual solution

# Create a list of feature indices
feature_indices = np.arange(len(solution_vector))

# Create a list indicating whether each feature is selected (1) or not selected (0)
selected_features = solution_vector

# Plot the selected features
plt.figure(figsize=(10, 4))
plt.bar(feature_indices, selected_features, color='b', align='center')
plt.xlabel('Feature Index')
plt.ylabel('Selected (1) / Not Selected (0)')
plt.title('Selected Features (1) vs. Not Selected Features (0)')
plt.xticks(feature_indices)
plt.show()
'''