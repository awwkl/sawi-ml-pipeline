import numpy as np
import glob
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.preprocessing import LabelBinarizer


def open(path):
    testFiles = path + "/test_set/totalFeatures5.csv"
    trainingFiles = glob.glob(path + "/training_set/*.csv")
    list_training = []  # training set list
    list_header = []  # get the list of headers of dfs for training & testing
    for file in trainingFiles:
        df = pd.read_csv(file, index_col=None, header=None, skiprows=0)
        # ########################
        # for i in df.index:
        #     row = df.iloc[i]      # get the i-th row of dataframe
        #     list1.append(df)      # merge the training files into a list
        # ##################### cannot not do this because different versions have different features
        head = df.iloc[0]
        head.tolist()
        list_header.append(head)  # list of header for training set
        list_training.append(df)  # list of training set

    testing = pd.read_csv(testFiles, index_col=None, header=None, skiprows=0)
    # training = pd.concat(list1, axis=0, ignore_index=True)
    list_header.append(testing.iloc[0].tolist())
    return testing, list_training, list_header


def df_get(df):
    """
    :param df: a data frame with header
    :return: get rid of the header of data frame
    """
    header = df.iloc[0]
    # Create a new variable called 'header' from the first row of the dataset

    # Replace the dataframe with a new one which does not contain the first row
    df = df[1:]

    # Rename the dataframe's column values with the header variable
    df = df.rename(columns=header)
    h1 = list(df.columns.values)  # get the value of header of the df
    return df, h1


def common_get(list_header):
    """
    :param list_header: list of training & testing headers
    :return: common header
    """

    golden_fea = ["F116", "F115", "F117", "F120", "F123", "F110", "F105", "F68", "F101", "F104", "F65", "F22",
                  " F94", "F71", "F72", "F25", "F3-", "F15", "F126", "F41", "F77"]
    golden_fea.append("category")
    golden_list = []
    count_list = []
    for header in list_header:
        golden = []
        count = 0
        for i in header:
            if i.startswith(tuple(golden_fea)):
                count += 1
                golden.append(i)
        # print("number of golden fea:", count)
        count_list.append(count)
        golden_list.append(golden)

    common = set(golden_list[0])
    for s in golden_list[1:]:
        common.intersection_update(s)
    return common


def trim(df, common):
    df1, header = df_get(df)
    for element in header:
        if element not in common:
            df1 = df1.drop(element, axis=1)
    df_trim = df1
    return df_trim


def intersect(a, b):
    """ return the intersection of two lists """
    return list(set(a) & set(b))


def merge1(testing, list1, list2):
    # not in use
    """
    :param testing: testing set
    :param list1: list of training data frame
    :param list2: list of headers of training set
    :return: get rid of uncommon parts of 5 version in training set
    """

    head5 = testing.iloc[0].tolist()
    holder = head5
    for i in list2:
        common = intersect(i, holder)
        holder = common

    common_header = common

    df, header = df_get(list1[0])  # ONLY USE THE VERSION 4 FOR TRAINING
    for element in header:
        if element not in common_header:
            df = df.drop(element, axis=1)
    df_merge = df

    # df_merge = pd.concat([i for i in list1], ignore_index=True, sort=True)

    return df_merge, common_header


def is_number(df):
    '''
    :param df: input should be training_x, testset_x(type: data frame)
    :return: return is index of numeric features
    '''

    index = []
    position = 0
    for i in range(len(df.iloc[0])):
        s = df.iloc[0, i]
        try:
            float(s)  # for int, long and float
            index.append(i)
        except ValueError:
            position += 1
    return index


def preprocess1(Y, X):
    index = []
    label = []
    for i in range(0, len(Y)):
        # y = Y[0][i]
        y = Y.iloc[i]
        if y == "close":
            # y = "yes"
            y = 1
        elif y == "open":
            # y = "no"
            y = 0
        elif y == "deleted":
            index.append(i)  # index is a list of index for deleted samples
        label.append(y)

    for i in sorted(index, reverse=True):  # delete samples with deleted label
        del label[i]
        del X[i]

    return label, X


def one_hot(df, index_num):
    """
    :param df: training_x or testset_x, type: data frame
    :param index_num: the index list of numerical features
    :return:
    """
    lb = LabelBinarizer()
    list_len = list(range(len(df.iloc[0])))
    index_onehot = list(set(list_len) - set(index_num))
    for i in index_onehot:
        df.iloc[:, i] = lb.fit_transform(df.iloc[:, 26]).tolist()
    return df


def data_clean(path, seed = 0):
    testing, list_training, list_header = open(path)
    common_header = common_get(list_header)
    np.random.seed(seed)
    # training set
    training_trim = trim(list_training[0], common_header)  # ONLY USE THE VERSION 4 FOR TRAINING

    # testing set
    testing_trim = trim(testing, common_header)

    # training set
    training_x = training_trim.iloc[:, :-1]
    training_y = training_trim.iloc[:, -1]

    # testing set
    testset_x = testing_trim.iloc[:, :-1]
    testset_y = testing_trim.iloc[:, -1]

    # remove the samples in training and test set with label "deleted"
    training_y, training_x = preprocess1(training_y, training_x)
    testset_y, testset_x = preprocess1(testset_y, testset_x)

    le = preprocessing.LabelEncoder()
    # kaggle's forums where to find valuable information

    # normalize the x for training and test sets
    min_max_scaler = preprocessing.MinMaxScaler()
    scaler = MinMaxScaler()

    testset_x = min_max_scaler.fit_transform(np.asarray(testset_x))
    training_x = min_max_scaler.fit_transform(np.asarray(training_x))

    return training_x, training_y, testset_x, testset_y
