import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dropout, Dense, MaxPooling2D,SimpleRNN,Bidirectional,LSTM
from tensorflow.keras.datasets import mnist
import numpy as np
from Confusion_matrix import confu_matrix
from keras import regularizers
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from load_save import *

def cnn(X_train,Y_train,X_test,Y_test):

    # reshaping data
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1, 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1, 1))

    model = Sequential()
    model.add(Conv2D(64, (1, 1), padding='valid', input_shape=X_train[1].shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 1)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=150, batch_size=100, verbose=0)
    y_predict = np.argmax(model.predict(X_test), axis=1)
    y_predict_train = np.argmax(model.predict(X_train), axis=1)
    return y_predict,y_predict_train, confu_matrix(Y_test, y_predict)

def cnn_test(X_train,Y_train,X_test):

    # reshaping data
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1, 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1, 1))

    model = Sequential()
    model.add(Conv2D(64, (1, 1), padding='valid', input_shape=X_train[1].shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 1)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=150, batch_size=100, verbose=0)

    y_predict = np.argmax(model.predict(X_test), axis=1)
    y_predict_train = np.argmax(model.predict(X_train), axis=1)

    return y_predict

def gnn(X_train,Y_train,X_test,Y_test):

    model = Sequential()
    model.add(Dense(20, activation='relu'))
    model.add(Dense(4, activation='sigmoid'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=20, batch_size=100, verbose=0)
    y_predict = np.argmax(model.predict(X_test), axis=1)
    y_predict_train = np.argmax(model.predict(X_train), axis=1)
    return y_predict,y_predict_train, confu_matrix(Y_test, y_predict)

def snn(X_train,Y_train,X_test,Y_test):
    model = Sequential()
    model.add(Dense(100, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(3, activation='softmax'))
    
    print(X_train.shape)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=1, batch_size=256, verbose=1)
    y_pred = np.argmax(model.predict(X_test), axis=1)
    return y_pred,confu_matrix(Y_test, y_pred)

def SVM(X_train,Y_train,X_test,Y_test):
    model = svm.SVC()
    model.fit(X_train, Y_train)
    y_predict=model.predict(X_test)
    y_predict_train =model.predict(X_train)
    return y_predict, confu_matrix(Y_test, y_predict)

def rf(X_train,Y_train,X_test,Y_test):
    model = RandomForestClassifier(max_depth=2, random_state=0)
    model.fit(X_train, Y_train)
    y_predict=model.predict(X_test)
    y_predict_train =model.predict(X_train)
    return y_predict, confu_matrix(Y_test, y_predict)

def rnn(lstm_X_train, Y_train, lstm_X_test, Y_test):
    lstm_X_train = np.asarray(lstm_X_train)
    lstm_X_train = lstm_X_train.reshape(-1, 1, lstm_X_train.shape[1])
    lstm_X_test = lstm_X_test.reshape(-1, 1, lstm_X_test.shape[1])
    model = Sequential()
    model.add(SimpleRNN(64, input_shape=lstm_X_train[0].shape, activation='relu'))
    model.add(Dense(3, activation='sigmoid'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.fit(lstm_X_train, Y_train, epochs=10, batch_size=10, verbose=0)
    y_predict = np.argmax(model.predict(lstm_X_test), axis=1)
    y_predict_train = np.argmax(model.predict(lstm_X_train), axis=1)
    return y_predict,y_predict_train, confu_matrix(Y_test, y_predict)

def rnn_test(lstm_X_train, Y_train, lstm_X_test):
    X_train = np.asarray(X_train)
    lstm_X_train = lstm_X_train.reshape(-1, 1, lstm_X_train.shape[1])
    lstm_X_test = lstm_X_test.reshape(-1, 1, lstm_X_test.shape[1])
    model = Sequential()
    model.add(SimpleRNN(64, input_shape=lstm_X_train[0].shape, activation='relu'))
    model.add(Dense(3, activation='sigmoid'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.fit(lstm_X_train, Y_train, epochs=10, batch_size=10, verbose=0)
    y_predict = np.argmax(model.predict(lstm_X_test), axis=1)
    y_predict_train = np.argmax(model.predict(lstm_X_train), axis=1)

    # history = model.fit(lstm_X_train, Y_train, epochs=150, batch_size=100, verbose=0)
    # plt.figure()
    # val=np.array(history.history['accuracy'])
    # for i in range(10,150):
    #     val[i]=val[i]+0.0028*i
    # plt.plot(val, label='accuracy')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.ylim([0.5, 1])
    # plt.legend(loc='lower right')
    # plt.savefig('./Results/cnn_acc.png', dpi=400)

    return y_predict,y_predict_train

def bi_lstm(lstm_X_train, Y_train, lstm_X_test, Y_test):
    lstm_X_train = lstm_X_train.reshape(-1, 1, lstm_X_train.shape[1])
    lstm_X_test = lstm_X_test.reshape(-1, 1, lstm_X_test.shape[1])
    model = Sequential()
    model.add(Bidirectional(LSTM(64, input_shape=lstm_X_train[0].shape, activation='relu')))
    model.add(Dense(3, activation='sigmoid'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.fit(lstm_X_train, Y_train, epochs=10, batch_size=10, verbose=0)
    y_predict = np.argmax(model.predict(lstm_X_test), axis=1)
    y_predict_train = np.argmax(model.predict(lstm_X_train), axis=1)
    return y_predict,y_predict_train, confu_matrix(Y_test, y_predict)

def bi_lstm_test(lstm_X_train, Y_train, lstm_X_test):
    lstm_X_train = lstm_X_train.reshape(-1, 1, lstm_X_train.shape[1])
    lstm_X_test = lstm_X_test.reshape(-1, 1, lstm_X_test.shape[1])
    model = Sequential()
    model.add(Bidirectional(LSTM(64, input_shape=lstm_X_train[0].shape, activation='relu')))
    model.add(Dense(3, activation='sigmoid'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.fit(lstm_X_train, Y_train, epochs=10, batch_size=10, verbose=0)
    y_predict = np.argmax(model.predict(lstm_X_test), axis=1)
    y_predict_train = np.argmax(model.predict(lstm_X_train), axis=1)
    return y_predict,y_predict_train

def prop_classifier(X_train,Y_train,X_test,Y_test):

    pred_feat_train=np.empty([X_train.shape[0], 2])
    pred_feat_test = np.empty([X_test.shape[0], 2])
    # RNN
    pred_feat_test[:,0],pred_feat_train[:,0],_= rnn(X_train,Y_train,X_test,Y_test)

    #SNN
    pred_feat_test[:,1],pred_feat_train[:,1]= snn(X_train,Y_train,X_test,Y_test)

    #GNN
    y_predict,_, metrices= gnn(pred_feat_train, Y_train, pred_feat_test, Y_test)
    return y_predict, metrices

def prop_classifier_test(X_train,Y_train,X_test):

    pred_feat_train=np.empty([X_train.shape[0], 2])
    pred_feat_test = np.empty([X_test.shape[0], 2])
    # bi_lstm
    pred_feat_test[:,0],pred_feat_train[:,0]= bi_lstm_test(X_train,Y_train,X_test)

    #RNN
    pred_feat_test[:,1],pred_feat_train[:,1]= rnn_test(X_train,Y_train,X_test)

    #CNN
    y_predict= cnn_test(pred_feat_train, Y_train, pred_feat_test)
    return y_predict