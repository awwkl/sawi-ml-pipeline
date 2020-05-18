import numpy as np
# import random
import statistics
from utils import Active
import time
# import warnings filter
from warnings import simplefilter

# ignore future warnings
simplefilter(action='ignore', category=FutureWarning)
import math
from sklearn import metrics


def active_learning(path, query='', stop='true', stopat=1, error='none', interval=100000, seed=0, initial=False):
    stopat = float(stopat)
    thres = 0
    starting = 1
    np.random.seed(seed)
    read = Active()
    read = read.create(path)
    read.interval = interval
    read.BM25(query.strip().split('_'))
    get_order = []
    num2 = read.get_allpos()
    target = int(num2 * stopat)

    if stop == 'est':
        read.enable_est = True
    else:
        read.enable_est = False

    total_recall = []
    rate = []

    while True:
        pos, neg, total, realpos = read.get_numbers()
        total_recall.append(pos / realpos)
        rate.append((pos + neg) / total)

        if pos + neg >= total:

            break

        if pos < starting or pos + neg < thres:
            if initial is False:
                for id in read.BM25_get():
                    read.code_error(id, error=error)
            else:
                for id in read.initial():
                    read.code_error(id, error=error)

        else:
            a, b, c, d = read.train(weighting=True, pne=True)
            if pos >= target:
                break
            if pos < 10:  # Uncertainity Sampling
                for id in a:
                    read.code_error(id, error=error)
            else:  # Certainity Sampling
                for id in c:
                    read.code_error(id, error=error)
                get_order = read.get_order()  # to get the auc

    print("total_recall:", total_recall)
    print("cost:", rate)
    print("total_recall achieved:", total_recall[-1])
    print("cost or percentage of instances having been retrived:", rate[-1])
    for i in range(len(get_order)):
        if get_order[i] != 0:
            get_order[i] = math.pow(10, 4) - get_order[i]

    testset_y, clf = read.loadfile()

    label = []
    for i in range(0, len(testset_y)):
        y = testset_y[i]
        if y == "yes":
            # y = "close"
            y = 1
        elif y == "no":
            # y = "open"
            y = 0
        label.append(y)

    auc = metrics.roc_auc_score(label, get_order)
    return read, rate[-1], auc, clf


if __name__ == "__main__":

    projects = ['derby', 'mvn', 'lucence', 'phoenix', 'cass', 'jmeter', 'tomcat', 'ant', 'commons']
    stopats = [0.7, 0.8, 0.9, 1]    # set the total recall threshold
    # stopats = [1]
    for stopat_id in stopats:
        print("----------threshold stop at----------:", stopat_id)
        for proj in projects:
            path = r'../data/total_features/' + proj
            print("-----------------" + proj + "----------------------")
            filename = path + "/test_set/totalFeatures5.csv"
            AUC = []
            cost = []
            repeated_times = 4
            for i in range(1, 1 + repeated_times):
                read, rate, auc, clf = active_learning(path, stopat=stopat_id,
                                                       seed=int(time.time() * 1000) % (2 ** 32 - 1),
                                                       initial=False)
                AUC.append(auc)
                cost.append(rate)
            print("classifier", clf)
            print("---------- total recall (threshold stop at)----------:", stopat_id)
            print("-----------------" + proj + "----------------------")
            print('AUC', AUC)
            print("cost", cost)
            AUC_med = statistics.median(AUC)
            AUC_iqr = np.subtract(*np.percentile(AUC, [75, 25]))
            COST_med = statistics.median(cost)
            COST_iqr = np.subtract(*np.percentile(cost, [75, 25]))
            print("AUC_median", AUC_med)
            print("AUC_iqr", AUC_iqr)
            print("COST_med", COST_med)
            print("COST_iqr", COST_iqr)
