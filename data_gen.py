import statistics
import numpy as np
import pandas as pd
from MRFO import OriginalMRFO
from load_save import *
from sklearn.model_selection import train_test_split

feat, label = [],[]

def feat_extract(data):
    feat = np.empty([data.shape[0], data.shape[1] + 3])
    feat[0:data.shape[0], 0:data.shape[1]] = data
    # Data Cleaning and dummy variable for nan
    data = np.nan_to_num(data)
    for i in range(data.shape[1]):
        # Standard Deviation
        feat[i, data.shape[1]] = statistics.stdev(data[i, :])
        feat[i, data.shape[1] + 1] = statistics.mean(data[i, :])
        feat[i, data.shape[1] + 2] = statistics.mode(data[i, :])

    # Normalization
    feat = feat / np.max(feat, axis=0)
    # Data Cleaning and dummy variable for nan
    feat = np.nan_to_num(feat)
    return feat


def datagen():
    # Dataset 1
    data = pd.read_csv('C:/Users/alosh/OneDrive/Desktop/VSCODE/DeepthyJ/Dataset/TimeBasedFeatures-10s-Layer2.csv').drop(labels=['Source IP', ' Destination IP'], axis=1)
    data=data.replace('AUDIO',0)
    data = data.replace('VOIP', 0)
    data = data.replace('BROWSING', 3)
    data = data.replace('FILE-TRANSFER', 1)
    data = data.replace('MAIL', 1)
    data = data.replace('P2P', 2)
    data = data.replace('VIDEO', 2)
    data = data.replace('CHAT', 3)
    data = pd.DataFrame.to_numpy(data)
    label.append(data[:,-1].astype('int16'))
    # get unique values and counts of each value
    unique, counts = np.unique(label[0], return_counts=True)
    data = np.delete(data, 26, axis=1)

    feat.append(feat_extract(data))

    # Dataset 2
    data = pd.read_csv('C:/Users/alosh/OneDrive/Desktop/VSCODE/DeepthyJ/Dataset/Darknet.csv', on_bad_lines='skip').drop(labels=['Flow ID', 'Src IP',
                                                                'Dst IP','Timestamp','Label1'], axis=1)
    data = data.replace('AUDIO-STREAMING', 0)
    data = data.replace('Audio-Streaming', 0)
    data = data.replace('VOIP', 0)
    data = data.replace('Browsing', 3)
    data = data.replace('File-Transfer', 1)
    data = data.replace('File-transfer', 1)
    data = data.replace('Email', 1)
    data = data.replace('P2P', 2)
    data = data.replace('Video-Streaming', 2)
    data = data.replace('Video-streaming', 2)
    data = data.replace('Chat', 3)
    data = pd.DataFrame.to_numpy(data)
    label.append(data[:, -1].astype('int16'))
    # get unique values and counts of each value
    unique, counts = np.unique(label[0], return_counts=True)
    data = np.delete(data, 26, axis=1)
    feat.append(feat_extract(data))


    # Train and test data split
    # split into train test sets
    train_data, test_data, train_lab, test_lab=[],[],[],[]
    k=[0.4, 0.3, 0.2]
    for i in range(2):
        train_data1, test_data1, train_lab1, test_lab1 = [], [], [], []
        for j in range(3):

            X_train, X_test, y_train, y_test = train_test_split(feat[i], label[i], test_size=k[j])

            train_data1.append(X_train)
            test_data1.append(X_test)
            train_lab1.append(y_train)
            test_lab1.append(y_test)
        train_data.append(train_data1)
        test_data.append(test_data1)
        train_lab.append(train_lab1)
        test_lab.append(test_lab1)
    save('X_train', train_data)
    save('X_test', test_data)
    save('y_train', train_lab)
    save('y_test', test_lab)

datagen()

X_train = load('X_train')

# For Dataset 1
num_features_dataset1 = feat[0].shape[1]  # Assuming feat[0] is the feature matrix for Dataset 1
print(f"Number of features in Dataset 1: {num_features_dataset1}")

# For Dataset 2
num_features_dataset2 = feat[1].shape[1]  # Assuming feat[1] is the feature matrix for Dataset 2
print(f"Number of features in Dataset 2: {num_features_dataset2}")
