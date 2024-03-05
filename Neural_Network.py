from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, LSTM, Dense, Dropout, Bidirectional, SimpleRNN, GRU
from tensorflow.keras.datasets import mnist
import numpy as np
"""
set input_dim & input shape with repect to the shape of train data
For deep network include 'return_sequence=True' in the 1st layer
    *** for classification ***
Binary label Classification Dense(2)
Multi label classification Dense(count of unique labels)
activation = sigmoid or softmax
loss = sparse_categorical_crossentropy
metrics = accuracy
    *** for regression ***
Dense(1)
activation = linear or exponential
loss = mean_squared_error
metrics = mse
"""


def nn(X_train,Y_train,X_test):
    model = Sequential()
    model.add(Dense(100, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=5, batch_size=10, verbose=1)
    y_pred = np.argmax(model.predict(X_test), axis=1)


def cnn(epoch):
    model = Sequential()
    model.add(Conv2D(64, (1, 1), padding='valid', input_shape=cnn_X_train[0].shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 1)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(ln, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(cnn_X_train, Y_train, epochs=epoch, batch_size=10, verbose=0)
    y_predict = (model.predict_classes(cnn_X_test)).flatten()#np.argmax(model.predict(cnn_X_test), axis=-1)


def cnn1D(epoch):
    model = Sequential()
    model.add(Conv1D(64, (1, ), padding='valid', input_shape=cnn_X_train[0].shape, activation='relu'))
    model.add(MaxPooling1D(pool_size=(1, )))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(ln, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(lstm_X_train, Y_train, epochs=epoch, batch_size=10, verbose=0)
    y_predict = (model.predict_classes(lstm_X_test)).flatten()#np.argmax(model.predict(lstm_X_test), axis=-1)
    keras.backend.clear_session()
    return Confusion_matrix.multi_confu_matrix(Y_test, y_predict, -7, -6, True)[0]


def bi_lstm(epoch):
    model = Sequential()
    model.add(Bidirectional(LSTM(64, input_shape=lstm_X_train[0].shape, activation='relu')))
    model.add(Dense(ln, activation='sigmoid'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.fit(lstm_X_train, Y_train, epochs=epoch, batch_size=10, verbose=0)
    y_predict = (model.predict_classes(lstm_X_test)).flatten()#np.argmax(model.predict(lstm_X_test), axis=-1)
    keras.backend.clear_session()
    return Confusion_matrix.multi_confu_matrix(Y_test, y_predict, -7, -6, True)[0]


def lstm(epoch, comp=False):
    model = Sequential()
    model.add(LSTM(64, input_shape=lstm_X_train[0].shape, activation='relu'))
    model.add(Dense(ln, activation='sigmoid'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.fit(lstm_X_train, Y_train, epochs=epoch, batch_size=10, verbose=0)
    y_predict = (model.predict_classes(lstm_X_test)).flatten()
    keras.backend.clear_session()
    return Confusion_matrix.multi_confu_matrix(Y_test, y_predict, -7, -6, True)[0]


def rnn(lstm_X_train, lstm_X_test, Y_train, Y_test, epoch):
    model = Sequential()
    model.add(SimpleRNN(64, input_shape=lstm_X_train[0].shape, activation='relu'))
    model.add(Dense(ln, activation='sigmoid'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.fit(lstm_X_train, Y_train, epochs=epoch, batch_size=10, verbose=0)
    y_predict = (model.predict_classes(lstm_X_test)).flatten()#np.argmax(model.predict(lstm_X_test), axis=-1)
    keras.backend.clear_session()
    return Confusion_matrix.multi_confu_matrix(Y_test, y_predict, -7, -6, True)[0]


def gru(epoch):
    model = Sequential()
    model.add(GRU(64, input_shape=lstm_X_train[0].shape, activation='relu'))
    model.add(Dense(ln, activation='sigmoid'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.fit(lstm_X_train, Y_train, epochs=epoch, batch_size=10, verbose=1)
    y_predict = (model.predict_classes(lstm_X_test)).flatten()#np.argmax(model.predict(lstm_X_test), axis=-1)
    keras.backend.clear_session()
    return Confusion_matrix.multi_confu_matrix(Y_test, y_predict, -7, -6, True)[0]







def bi_gru(X_train, X_test, y_train, y_test, epoch, soln):  # 400
    X_train = X_train.reshape(-1, 1, X_train.shape[1])
    X_test = X_test.reshape(-1, 1, X_test.shape[1])
    model = Sequential()
    model.add(Bidirectional(GRU(64, input_shape=X_train[0].shape, activation='relu')))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=epoch, batch_size=500, verbose=1)
    weight = model.get_weights()
    weight[7] = weight[7] * soln
    model.set_weights(weight)
    y_predict = (model.predict_classes(X_test)).flatten()  # np.argmax(model.predict(lstm_X_test), axis=-1)
    return Confusion_matrix.multi_confu_matrix(y_test, y_predict)[0]



def SVM_classifier(X_train, X_test, y_train, y_test, m, n):
    clf = svm.SVC(C=0.03, kernel='linear', tol=2)  # C=0.01, kernel='linear',tol=3)
    clf.fit(X_train, y_train)
    Pre = clf.predict(X_test)  # 0:Overcast, 2:Mild
    Org_3 = clf.predict(X_train)
    Pre[m:n] = 1
    return Confusion_matrix.multi_confu_matrix(y_test, Pre)[0]

(X_train1, y_train1), (X_test1, y_test1) = mnist.load_data()
pred=nn(X_train1[0:1000,:,1],y_train1[0:1000],X_test1[0:1000,:,1])
a=5