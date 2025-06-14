import numpy as np
from sklearn.metrics import confusion_matrix

# Set the size of your random dataset
data_size = 10000000

# Generate random labels for the true values (ground truth)
y_true = np.random.randint(0, 4, size=data_size)  # Assuming 4 classes, adjust as needed

# Generate random labels for the predicted values
y_pred = np.random.randint(0, 4, size=data_size)

# Compute the confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# Print or visualize the confusion matrix
print("Confusion Matrix:")
print(conf_matrix)