import numpy as np
import glob
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn import tree
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from collections import defaultdict
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
import statistics
import time
import warnings
import cleaning

""" NOTES:
      - requires Python 3.0 or greater
      - order of the original lists is not preserved
"""


def main(path, stop_at, clf, seed=0):

    training_x, training_y, testset_x, testset_y = cleaning.data_clean(path, seed)
    # start_time = time.time()
    clf.fit(training_x, training_y)
    # print("running time", time.time() - start_time)

    # pos_at = list(clf.classes_).index("yes")
    pos_at = list(clf.classes_).index(1)

    prob = clf.predict_proba(testset_x)[:, pos_at]

    auc = metrics.roc_auc_score(testset_y, prob)

    sorted_label = []
    order = np.argsort(prob)[::-1][:]  # numpy.ndarray
    # pos_all = sum([1 for label_real in testset_y if label_real == "yes"])
    pos_all = sum([1 for label_real in testset_y if label_real == 1])
    num_all = sum([1 for label_real in testset_y])
    print("number of samples:", num_all)
    total_recall = []
    length = []
    for i in order:
        a = testset_y[i]  # real label
        sorted_label.append(a)
        # pos_get = sum([1 for label_real in sorted_label if label_real == "yes"])
        pos_get = sum([1 for label_real in sorted_label if label_real == 1])
        length.append(len(sorted_label) / num_all)
        total_recall.append(pos_get / pos_all)
        # print(pos_get, len(sorted_label))
# ######
    total_recall = total_recall[::10]
    rate = length[::10]
    # append(1) in case that list out of range
    total_recall.append(1)
    rate.append(1)

    if type(stop_at) is tuple:
        stop_at = stop_at[0]

    stop = 0
    for index in range(len(total_recall)):
        if total_recall[index] >= stop_at:
            stop = index
            break

    # AUC score in Table six
    print("AUC", auc)
    print("pos_get", pos_get)
    # ########
    # RQ 3 in Figure 4: test results
    # print("total_recall_baseline", total_recall)
    # print("rate_baseline", rate)
    # ########
    print("total recall stop_at", total_recall[stop])
    # print("total recall stop_before", total_recall[stop - 1])
    return rate[stop], auc


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # ############################################################
    # weighting = True
    # clf = svm.SVC(kernel='linear', probability=True, class_weight='balanced') if weighting else svm.SVC(
    #     kernel='linear', probability=True)
    clf1 = svm.SVC(kernel='linear', probability=True, class_weight='balanced')
    clf2 = tree.DecisionTreeClassifier()
    clf3 = RandomForestClassifier()
    clf_list = [clf1, clf2, clf3]
    projects = ['derby', 'mvn', 'lucence', 'phoenix', 'cass', 'jmeter', 'tomcat', 'ant', 'commons']
    stopats = [1]
    # stopats = [0.7, 0.8, 0.9, 1]

    for clf in clf_list:
        print("classifier:", clf)
        for stopat_id in stopats:
            print("----------threshold stop at----------:", stopat_id)
            for project in projects:
                path = r'../data/total_features/' + project
                print("-----------------" + project + "----------------------")
                AUC = []
                cost = []
                repeated_times = 10
                for i in range(1, 1+repeated_times):
                    rate, auc = main(path, stop_at=stopat_id,
                                     seed=int(time.time() * 1000) % (2 ** 32 - 1), clf=clf)
                    AUC.append(auc)
                    cost.append(rate)
                AUC_med = statistics.median(AUC)
                AUC_iqr = np.subtract(*np.percentile(AUC, [75, 25]))
                COST_med = statistics.median(cost)
                COST_iqr = np.subtract(*np.percentile(cost, [75, 25]))
                print("----------threshold stop at----------:", stopat_id)
                print("-----------------" + project + "----------------------")
                # AUC score in Table six
                print('AUC', AUC)
                print('cost', cost)
                print("AUC_median", AUC_med)
                print("AUC_iqr", AUC_iqr)
                print("COST_med", COST_med)
                print("COST_iqr", COST_iqr)
