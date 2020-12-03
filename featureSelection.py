
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif


class featutreSelection():
    def  quasi_constant(data, percentage):
        quasi_constant_feat = []
        for feature in data.columns:
            predominant = (data[feature].value_counts() / np.float(len(data))).sort_values(ascending=False).values[0]
            if predominant > percentage:
                quasi_constant_feat.append(feature)
        return quasi_constant_feat

    def duplicateColumns(data):
        duplicated_feat_pairs = {}
        _duplicated_feat = []

        for i in range(0, len(data.columns)):
            if i % 5 == 0:
                print(i)
            feat_1 = data.columns[i]
            if feat_1 not in _duplicated_feat:
                duplicated_feat_pairs[feat_1] = []

                for feat_2 in data.columns[i + 1:]:
                    if data[feat_1].equals(data[feat_2]):
                        duplicated_feat_pairs[feat_1].append(feat_2)
                        _duplicated_feat.append(feat_2)
        print('--------')
        return _duplicated_feat

    def correlation(dataset, threshold):
        col_corr = set()
        corr_matrix = dataset.corr()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > threshold: 
                    print(abs(corr_matrix.iloc[i, j]), corr_matrix.columns[i], corr_matrix.columns[j])
                    colname = corr_matrix.columns[j]
                    col_corr.add(colname)
        return col_corr

    def mutualInformation(X_train, y_train):
        mi = mutual_info_classif(X_train, y_train)
        mi = pd.Series(mi)
        mi.index = X_train.columns
        mi.sort_values(ascending=False).plot.bar(figsize=(20, 6))
        plt.ylabel('Mutual Information')

