from __future__ import division
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from sklearn import svm
import time
import pandas as pd
import chardet
import os
import glob
import pandas
import cleaning
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn import svm


class Active(object):
    def __init__(self):
        self.fea_num = 4000
        self.step = 10  # update your model after a specific step
        self.enough = 30  # a threshold to convert to aggressive undersampling
        self.atleast = 100  # assume all unlabeled data as negative
        # self.enable_est = True
        self.interval = 500000000
        self.true_count = 0
        self.false_count = 0
        self.count = 0  # count for order in auc

    def create(self, path, dataset=None):
        self.path = path
        self.name = self.path.split(".")[0]
        # self.flag = True
        self.hasLabel = True
        self.record = {"x": [], "pos": []}
        self.body = {}
        self.round = 0
        self.dataset = dataset
        self.reorder = []  # record the retrieving order
        try:
            self.loadfile()
            # print("Preprocessing")
            self.preprocess()
            # self.save()
        except:
            ## file not found in workspace ##
            print("Data file not found")
            self.flag = False

        return self

    def binary_label(self, Y):
        label = []
        for i in range(0, len(Y)):
            # y = Y[0][i]
            y = Y[i]
            if y == 1:
                # y = "close"
                y = "yes"
            elif y == 0:
                # y = "open"
                y = "no"
            label.append(y)
        return label

    def loadfile(self):

        training_x, training_y, testset_x, testset_y = cleaning.data_clean(self.path)
        training_y, testset_y = self.binary_label(training_y), self.binary_label(testset_y)

        self.clf1 = tree.DecisionTreeClassifier()
        # self.clf1 = RandomForestClassifier()
        self.clf1.fit(training_x, training_y)

        content = testset_x
        label = testset_y

        fields = ["text features"]
        header = ["text features", "label"]

        self.body["label"] = [c for c in label[0:]]
        for field in fields:
            ind = header.index(field)  # ind = 4
            self.body[field] = content
            print("length of samples in version 5(deleted removed)", len(self.body[field]))

        try:
            ind = header.index("code")
            self.body["code"] = [c[ind] for c in content[0:]]

        except:
            self.body["code"] = ['undetermined'] * (len(content))

        try:
            ind = header.index("time")
            self.body["time"] = [c[ind] for c in content[0:]]

        except:
            self.body["time"] = [0] * (len(content))
            # get the exact order of samples retrieved

        try:
            ind = header.index("syn_error")
            self.body["syn_error"] = [c[ind] for c in content[0:]]

        except:
            self.body["syn_error"] = [0] * (len(content))

        try:
            ind = header.index("fixed")
            self.body["fixed"] = [c[ind] for c in content[0:]]

        except:
            self.body["fixed"] = [0] * (len(content))

        try:
            ind = header.index("count")
            self.body["count"] = [c[ind] for c in content[0:]]

        except:
            self.body["count"] = [0] * (len(content))

        try:
            ind = header.index("retrieve")
            self.body["retrieve"] = [c[ind] for c in content[0:]]
        except:
            self.body["retrieve"] = [0] * (len(content))

        return label, self.clf1

    def get_numbers(self):
        total = len(self.body["code"])
        realpos = Counter(self.body["label"])["yes"]
        pos = Counter(self.body["code"])["yes"]
        neg = Counter(self.body["code"])["no"]
        try:
            tmp = self.record['x'][-1]
        except:
            tmp = -1
        if int(pos + neg) > tmp:
            self.record['x'].append(int(pos + neg))
            self.record['pos'].append(int(pos))
        # ---Sherry delete above--------
        self.pool = np.where(np.array(self.body['code']) == "undetermined")[0]
        self.labeled = list(set(range(len(self.body['code']))) - set(self.pool))
        return pos, neg, total, realpos

    def preprocess(self):

        self.csr_mat = self.body["text features"]
        return

    def initial(self):
        pos_at = list(self.clf1.classes_).index("yes")
        self.prob1 = self.clf1.predict_proba(self.csr_mat[self.pool])[:, pos_at]
        ids = self.pool[np.argsort(self.prob1[self.pool])[::-1][:self.step]]  # index of prob1 in descending order
        return ids

    ## BM25 ##
    def BM25(self, query):
        if query[0] == '':
            self.bm = np.random.rand(len(self.body["label"]))

            return
        b = 0.75
        k1 = 1.5

        content = [self.body["text features"][index] for index in
                   range(len(self.body["label"]))]
        #######################################################
        # self.pool is a list of candidate indexs
        # self.bm contains the candidate value. The larger, the higher priority to be selected
        # argsort return a 1-dim array of indexes. The first item contains the index of the smallest value of self.bm. The last ite
        # argsort()[::-1] will inverse the order.
        # plus [:self.step] is to select the most #step similar documents to a specific query

        tfidfer = TfidfVectorizer(lowercase=True, stop_words="english", norm=None, use_idf=False, smooth_idf=False,
                                  sublinear_tf=False, decode_error="ignore")
        tf = tfidfer.fit_transform(content)
        d_avg = np.mean(np.sum(tf, axis=1))
        score = {}
        for word in query:
            score[word] = []
            try:
                id = tfidfer.vocabulary_[word]
            except:
                score[word] = [0] * len(content)
                continue
            df = sum([1 for wc in tf[:, id] if wc > 0])
            idf = np.log((len(content) - df + 0.5) / (df + 0.5))
            for i in range(len(content)):
                score[word].append(
                    idf * tf[i, id] / (tf[i, id] + k1 * ((1 - b) + b * np.sum(tf[0], axis=1)[0, 0] / d_avg)))
        self.bm = np.sum(list(score.values()), axis=0)

    # BM25_get is to get the indexes of bm at indexes of pool, then reverse and take the first step size of them
    def BM25_get(self):
        return self.pool[np.argsort(self.bm[self.pool])[::-1][:self.step]]

    # train for baseline
    # def train_bl(self, weighting=True):
    #     clf = svm.SVC(kernel='linear', probability=True, class_weight='balanced') if weighting else svm.SVC(
    #         kernel='linear', probability=True)

    # # Train model ##
    def train(self, pne=True, weighting=True):
        # clf = svm.SVC(kernel='linear', probability=True, class_weight='balanced') if weighting else svm.SVC(
        #     kernel='linear', probability=True)

        clf = tree.DecisionTreeClassifier()
        # clf = RandomForestClassifier()
        poses = np.where(np.array(self.body['code']) == "yes")[0]
        negs = np.where(np.array(self.body['code']) == "no")[0]
        left = poses
        decayed = list(left) + list(negs)
        unlabeled = np.where(np.array(self.body['code']) == "undetermined")[0]
        try:
            unlabeled = np.random.choice(unlabeled, size=np.max((len(left), self.atleast)), replace=False)

        except:
            pass

        # PRESUMTIVE NON RELEVANT AFTER APPLYING BM25
        # Examples Presume all examples are false, because true examples are few
        # This reduces the biasness of not doing random sampling
        if not pne:
            unlabeled = []

        labels = np.array([x if x != 'undetermined' else 'no' for x in self.body['code']])
        all_neg = list(negs) + list(unlabeled)
        sample = list(decayed) + list(unlabeled)
        clf.fit(self.csr_mat[sample], labels[sample])

        """"""
        pos_at = list(clf.classes_).index("yes")
        prob = np.array(clf.predict_proba(self.csr_mat)[:, pos_at])
        """"""

        # # aggressive undersampling ##
        if len(poses) >= self.enough:

            """"""  # ########change classifiers
            # train_dist = clf.decision_function(self.csr_mat[all_neg])
            # pos_at = list(clf.classes_).index("yes")
            # if pos_at:
            #     train_dist = -train_dist
            """"""

            negs_sel = np.argsort(prob[all_neg])[:len(left)]
            sample = list(left) + list(np.array(all_neg)[negs_sel])
            clf.fit(self.csr_mat[sample], labels[sample])
        elif pne:

            """"""  # ########change classifiers
            # train_dist = clf.decision_function(self.csr_mat[unlabeled])
            # pos_at = list(clf.classes_).index("yes")
            # if pos_at:
            #     train_dist = -train_dist
            """"""

            unlabel_sel = np.argsort(prob[unlabeled])[:int(len(unlabeled) / 2)]
            sample = list(decayed) + list(np.array(unlabeled)[unlabel_sel])
            clf.fit(self.csr_mat[sample], labels[sample])

        # # correct errors with human-machine disagreements ##
        if self.round == self.interval:
            self.round = 0
            susp, conf = self.susp(clf)
            return susp, conf, susp, conf
        else:
            self.round = self.round + 1
        #####################################################
        uncertain_id, uncertain_prob = self.uncertain(clf)
        certain_id, certain_prob = self.certain(clf)
        if self.enable_est:
            if self.last_pos > 0 and len(poses) - self.last_pos > 0:
                self.est_num, self.est = self.estimate_curve(clf, reuse=True, num_neg=len(sample) - len(left))
            else:
                self.est_num, self.est = self.estimate_curve(clf, reuse=False, num_neg=len(sample) - len(left))
            return uncertain_id, self.est[uncertain_id], certain_id, self.est[certain_id]
        else:
            return uncertain_id, uncertain_prob, certain_id, certain_prob

    ## Get certain ##
    def certain(self, clf):
        pos_at = list(clf.classes_).index("yes")
        prob = clf.predict_proba(self.csr_mat[self.pool])[:, pos_at]
        order = np.argsort(prob)[::-1][:self.step]  # reserves the array and return the last step elements
        return np.array(self.pool)[order], np.array(prob)[order]

    ## Get uncertain ##
    def uncertain(self, clf):
        pos_at = list(clf.classes_).index("yes")
        prob = clf.predict_proba(self.csr_mat[self.pool])[:, pos_at]

        """"""  # ########change classifiers
        # train_dist = clf.decision_function(self.csr_mat[self.pool])
        """
        order = np.argsort(np.abs(train_dist))[:self.step]  ## uncertainty sampling by distance to decision plane
        """
        """"""
        order = np.argsort(np.abs(prob - 0.5))[:self.step]  ## uncertainty sampling via probability prediction
        return np.array(self.pool)[order], np.array(prob)[order]

    def code(self, id, label):
        self.body["code"][id] = label
        self.body["time"][id] = time.time()
        self.count = self.count + 1
        self.body["retrieve"][id] = self.count

    def get_order(self):
        return self.body["retrieve"]

    def code_error(self, id, error='none'):
        # simulate a human reader
        if error == 'circle':
            self.code_circle(id, self.body['label'][id])
        elif error == 'random':
            self.code_random(id, self.body['label'][id])
        elif error == 'three':
            self.code_three(id, self.body['label'][id])
        else:
            self.code(id, self.body['label'][id])

    def get_allpos(self):
        return len([1 for c in self.body["label"] if c == "yes"])


