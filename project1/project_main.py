"""
Project 1 Isolated digit recognition in noisy environments

Author : Seunghyeon Shin (2022325988)

"""

import os
import glob
import pickle
import utils

# data prepare
utils.dataset_generation(tag='extracted_feature', base_dir='./dataset/', n_mfcc=20, normalize=True)
query = os.path.abspath("{base}/*".format(base='./dataset/extracted_feature/'))
files = sorted(glob.glob(query))
files = [f for f in files if os.path.isfile(f)]
features_train = []
features_val = []
features_test = []
label_train = []
label_val = []
label_test = []

for idx, file in enumerate(files):
    label = file[-5]
    with open(file, 'rb') as f:
        data = pickle.load(f)

    if 'train' in file:
        features_train.append(data)
        label = [int(label)] * len(data)
        label_train.append(label)

    elif 'val' in file:
        features_val.append(data)
        label = [int(label)] * len(data)
        label_val.append(label)

    elif 'test' in file:
        features_test.append(data)
        label = [int(label)] * len(data)
        label_test.append(label)


for i in range(len(features_train)):
    if i == 0:
        data_tr = features_train[i]
        data_val = features_val[i]
        data_test = features_test[i]
        ans_tr = label_train[i]
        ans_val = label_val[i]
        ans_test = label_test[i]
    else:
        data_tr = data_tr + features_train[i]
        data_val = data_val + features_val[i]
        data_test = data_test + features_test[i]
        ans_tr = ans_tr + label_train[i]
        ans_val = ans_val + label_val[i]
        ans_test = ans_test + label_test[i]


utils.hmm_model(data_tr, ans_tr, data_val, ans_val, data_test, ans_test)
