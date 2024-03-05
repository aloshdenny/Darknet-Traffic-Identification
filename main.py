from objective_function import obj_fun
from MRFO import OriginalMRFO, IMRFO
from load_save import save, load
from BWO import OriginalBWO
from plot_res import plot_res
from classifier import *
import numpy as np

def full_analysis():
    X_train = load('X_train')
    X_test = load('X_test')
    y_train = load('y_train')
    y_test = load('y_test')

    # Optimization parameters
    lb = (np.zeros([1, X_train[0].shape[1]]).astype('int16')).tolist()[0]
    ub = (np.ones([1, X_train[0].shape[1]]).astype('int16')).tolist()[0]
    problem_dict1 = {"fit_func": obj_fun,
                     "lb": lb,
                     "ub": ub,
                     "minmax": "min"}
    epoch = 10
    pop_size = 10
    pro, cnn_val, ann_val, rf_val=[], [], [], []
    for i in range(2):
        for j in range(3):
            save('cur_X_train', X_train[i][j])
            save('cur_X_test', X_test[i][j])
            save('cur_y_train', y_train[i][j])
            save('cur_y_test', y_test[i][j])

            # Proposed
            model = IMRFO(epoch, pop_size)
            best_position, best_fitness = model.solve(problem_dict1)

            # MRFO
            model = OriginalMRFO(epoch, pop_size)
            best_position, best_fitness = model.solve(problem_dict1)

            # BWO
            model = OriginalBWO(epoch, pop_size)
            best_position, best_fitness = model.solve(problem_dict1)

            # CNN
            pred, met, met_train = cnn(X_train[i], y_train[i], X_test[i], y_test[i])
            cnn_val.append([met, met_train])


            # RF
            pred, met, met_train = rf(X_train[i], y_train[i], X_test[i], y_test[i])
            rf_val.append([met, met_train])

an=0
if an==1:
    full_analysis()

plot_res()