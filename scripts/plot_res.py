import numpy as np
import pandas as pd
from load_save import *
import matplotlib.pyplot as plt
from sklearn import metrics

def bar_plot(label, data1, data2, data3, metric, j):

    # create data
    df = pd.DataFrame([data1, data2, data3],
                      columns=label)
    df1 = pd.DataFrame()
    df1['Train-Test Split (%)'] = [60, 70, 80]
    df = pd.concat((df1, df), axis=1)
    # plot grouped bar chart
    df.plot(x='Train-Test Split (%)',
            kind='bar',
            stacked=False)

    plt.ylabel(metric)
    plt.legend(loc='lower right')
    plt.savefig('C:/Users/alosh/OneDrive/Desktop/VSCODE/DeepthyJ/Results/'+metric+str(j)+'.png', dpi=400)
    plt.show(block=False)


def evaluate(X_train, y_train, X_test, y_test,soln):
    met_train, met_test = cnn_w_test(X_train, y_train, X_test, y_test, soln)
    return np.array(met_train), np.array(met_test)

def plot_res():


    X_train = load('X_train')
    X_test = load('X_test')
    y_train = load('y_train')
    y_test = load('y_test')


    an = 0
    if an == 1:
        met_ann = load('ann_val')
        met_cnn = load('cnn_val')
        met_rf = load('rf_val')
        pro = load('pro')
        metrices, metrices_train=[], []
        for i in range(3):
            metrices1=np.empty([9,4])
            metrices_train1 = np.empty([9, 4])
            # ANN
            metrices_train1[:,0],metrices1[:,0]=met_ann[i][1], met_ann[i][0]

            # CNN
            metrices_train1[:,1], metrices1[:, 1] = met_cnn[i][1], met_cnn[i][0]

            # RF
            metrices_train1[:,2], metrices1[:, 2] = met_rf[i][1], met_rf[i][0]

            # Pro
            metrices_train1[:,3], metrices1[:, 3] = evaluate(X_train[i], y_train[i], X_test[i], y_test[i], pro[i])

            metrices.append(metrices1)
            metrices_train.append(metrices_train1)
        # save('metrices', metrices)
        # save('metrices_train', metrices_train)
    for j in range(2):
        cmp_met = load('cmp_met_updated')
        if j==0:
            metrices=load('metrices1_updated')
        else:
            metrices = load('metrices2_updated')
        mthod=['MRFO', 'BWO', 'RF', 'Proposed']
        metrices_plot=['Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F-Measure', 'MCC', 'NPV', 'FPR', 'FNR']

        # Bar plot
        for i in range(len(metrices_plot)):
            bar_plot(mthod, metrices[0][i,:], metrices[1][i,:], metrices[2][i,:], metrices_plot[i], j)

        for i in range(3):
            # Table
            print('Testing Metrices-'+ str(i+1)+'-Dataset- '+ str(j+1))
            tab=pd.DataFrame(metrices[i], index=metrices_plot, columns=mthod)
            print(tab)

        mthod2=['CNN', 'RNN', 'LSTM', 'SVM', 'NB', 'RF', 'Proposed']
        # Table
        print('Comparison Metrices-Dataset- ' + str(j + 1))
        tab = pd.DataFrame(cmp_met[j], index=metrices_plot, columns=mthod2)
        tab.to_csv('C:/Users/alosh/OneDrive/Desktop/VSCODE/DeepthyJ/Results/Comparison Metrices'+ str(j)+'.csv')
        print(tab)

        # horizontal Bar graph
        # creating the bar plot
        labels=['Delay-sensitive', 'Loss-sensitive', 'Bandwidth-sensitive', 'Best-effort']
        fig = plt.figure(figsize=(9,5))
        plt.barh(labels, [2862, 1216, 1987, 1979], color='maroon')
        plt.xlabel("Predicted")
        plt.ylabel("Classes")
        plt.savefig('C:/Users/alosh/OneDrive/Desktop/VSCODE/DeepthyJ/Results/traffic_classification.png', dpi=400)

    plt.show()
