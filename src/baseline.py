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
import sys
from pathlib import Path
import lime
import lime.lime_tabular
from lime import submodular_pick

""" NOTES:
      - requires Python 3.0 or greater
      - order of the original lists is not preserved
"""


def main(path, stop_at, clf, seed=0):

    training_x, training_y, testset_x, testset_y = cleaning.data_clean(path, seed)

    print("training_x:", training_x.shape)
    print("training_y:", len(training_y))
    print("testset_x:", testset_x.shape)
    print("testset_y:", len(testset_y))

    training_x.to_csv(path + "training_x.csv")
    testset_x.to_csv(path + "testset_x.csv")
    
    clf.fit(training_x, training_y)
    y_pred = clf.predict(testset_x)

    print(metrics.classification_report(testset_y, y_pred))
    print("accuracy:", metrics.accuracy_score(testset_y, y_pred))
    tn, fp, fn, tp = metrics.confusion_matrix(testset_y, y_pred).ravel()
    print("@@@ tn: {}, fp: {}, fn: {}, tp: {}".format(tn, fp, fn, tp))

    print("@@@ LIME - Creating explainer", flush=True)
    feature_names =  training_x.columns.values.tolist()
    explainer = lime.lime_tabular.LimeTabularExplainer(np.asarray(training_x), feature_names=feature_names, discretize_continuous=True)

    print("@@@ LIME - Random Sampling of Instances", flush=True)
    Path(path + "lime-random/").mkdir(parents=True, exist_ok=True)
    for iter in range(10):
        sample_no = np.random.randint(0, testset_x.shape[0])
        print("iter: %d, sample_no: %d, actual label: %s, predicted: %s" % (iter, sample_no, testset_y[sample_no], y_pred[sample_no]))
        exp = explainer.explain_instance(testset_x.iloc[sample_no], clf.predict_proba, num_features=10)
        exp.save_to_file(path + "lime-random/" + 'lime_random_' + str(iter) + '.html')

    print("@@@ LIME - Submodular Pick", flush=True)
    Path(path + "lime-sp/").mkdir(parents=True, exist_ok=True)
    sp_obj = submodular_pick.SubmodularPick(explainer, np.asarray(training_x), clf.predict_proba, sample_size=100, num_features=10, num_exps_desired=10)
    for iter in range(len(sp_obj.sp_explanations)):
        exp = sp_obj.sp_explanations[iter]
        exp.save_to_file(path + "lime-sp/" + 'lime_sp_obj_' + str(iter) + '.html')

    print("@@@ LIME - Investigating interesting instances: predicted differs from actual label", flush=True)
    Path(path + "lime-differs/").mkdir(parents=True, exist_ok=True)
    df_pred_and_actual = pd.DataFrame({ 'y_pred': y_pred, 'testset_y': testset_y })
    differs_list = df_pred_and_actual.index[ df_pred_and_actual['y_pred'] != df_pred_and_actual['testset_y'] ].tolist()
    print("Samples where predicted and actual label differs:", differs_list)

    for iter, sample_no in enumerate(differs_list):
        print("iter: %d, sample_no: %d, actual label: %s, predicted: %s" % (iter, sample_no, testset_y[sample_no], y_pred[sample_no]), flush=True)
        exp = explainer.explain_instance(testset_x.iloc[sample_no], clf.predict_proba, num_features=10)
        exp.save_to_file(path + "lime-differs/" + 'lime_differs_' + str(iter) + '.html')

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

    clf1 = svm.SVC(kernel='linear', probability=True, class_weight='balanced')
    clf2 = tree.DecisionTreeClassifier()
    clf3 = RandomForestClassifier()
    clf_list = [clf1]
    stopats = [1]
    # stopats = [0.7, 0.8, 0.9, 1]

    path = r'../data/current/'
    sys.stdout = open(path + 'stdout.txt', 'w')

    for clf in clf_list:
        print("@@@ A - classifier:", clf)
        for stopat_id in stopats:
            print("@@@ B - threshold stop at:", stopat_id)

            AUC = []
            cost = []
            repeated_times = 1
            for i in range(1, 1+repeated_times):
                print("@@@ C - Repeat number:", i)
                rate, auc = main(path, stop_at=stopat_id,
                                    seed=int(time.time() * 1000) % (2 ** 32 - 1), clf=clf)
                AUC.append(auc)
                cost.append(rate)
            AUC_med = statistics.median(AUC)
            AUC_iqr = np.subtract(*np.percentile(AUC, [75, 25]))
            COST_med = statistics.median(cost)
            COST_iqr = np.subtract(*np.percentile(cost, [75, 25]))
            print("----------threshold stop at----------:", stopat_id)
            # AUC score in Table six
            print('AUC', AUC)
            print('cost', cost)
            print("AUC_median", AUC_med)
            print("AUC_iqr", AUC_iqr)
            print("COST_med", COST_med)
            print("COST_iqr", COST_iqr)