import matplotlib.pyplot as plt
import numpy as np
import re


def draw(plt, values, type, line_style, color, label):
    plt.plot(np.arange(len(values)), values, type, linestyle=line_style, color=color, label=label)


if __name__ == '__main__':
    file_names = ['vgg_16_reduced.log', 'inception_bn.log']
    types = ['-', 'x']


    plt.figure(figsize=(8, 6))
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")

    for i, file_name in enumerate(file_names):

        log = open(file_name).read()
        log_tr = re.compile('.*Epoch\[(\d+)\].*Batch \[(\d+)\].*Train-rmse=([-+]?\d*\.\d+|\d+)').findall(log)
        log_va = re.compile('.*Epoch\[(\d+)\].*Validation-rmse=([-+]?\d*\.\d+|\d+)').findall(log)
        log_n_tr = re.compile('.*Epoch\[(\d+)\].*Batch \[(\d+)\].*Train-NRMSE=([-+]?\d*\.\d+|\d+)').findall(log)
        log_n_va = re.compile('.*Epoch\[(\d+)\].*Validation-NRMSE=([-+]?\d*\.\d+|\d+)').findall(log)

        log_tr = np.array(log_tr)
        log_n_tr = np.array(log_n_tr)

        data = {}
        for epoch, batch, rmse in log_tr:
            if len(data) == 0 or int(epoch) is not data[len(data) - 1][0]:
                data[len(data)] = [int(epoch), float(rmse), 1]
            else:
                data[len(data) - 1][1] += float(rmse)
                data[len(data) - 1][2] += 1
        tr_value = []
        for vals in data:
            tr_value.append(data[vals][1] / data[vals][2])

        data = {}
        for epoch, batch, rmse in log_n_tr:
            if len(data) == 0 or int(epoch) is not data[len(data) - 1][0]:
                data[len(data)] = [int(epoch), float(rmse), 1]
            else:
                data[len(data) - 1][1] += float(rmse)
                data[len(data) - 1][2] += 1
        n_tr_value = []
        for vals in data:
            n_tr_value.append(data[vals][1] / data[vals][2])


        idx = np.arange(len(tr_value))

        va_value = []
        for vals in log_va:
            va_value.append(vals[1])

        n_va_value = []
        for vals in log_n_va:
            n_va_value.append(vals[1])    


        draw(plt, tr_value, types[i], '-', 'r', "Train-RMSE/image size, " + file_name)
        draw(plt, va_value, types[i], '-', 'b', "Validation-RMSE/image size, " + file_name)
        draw(plt, n_tr_value, types[i], '--', 'r', "Train-RMSE/iod, " + file_name)
        draw(plt, n_va_value, types[i], '--', 'b', "Validation-RMSE/iod, " + file_name)

    plt.legend(loc="best")

    plt.yticks(np.arange(0, 0.2, 0.01))
    plt.ylim([0,0.2])
    plt.show()