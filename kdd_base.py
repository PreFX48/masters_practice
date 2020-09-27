import numpy as np
import pandas as pd
from tensorflow import keras

import re

def to_float_str(element):
    try:
        return str(float(element))
    except ValueError:
        return element



def get_dataset(target):
    targets = pd.read_csv(f'kdd/{target}.labels', names=['target'])['target']
    targets = targets.apply(lambda x: 1 if x == 1 else 0)
    data = pd.read_csv('kdd/kdd.data', sep='\t')
    data[target] = targets
    targets = data[target]
    data.drop([target], axis=1, inplace=True)
    
    categorical_features = {
        190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 209,
        210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228,
    }

    for i in categorical_features:
        data[data.columns[i]].fillna("?", inplace=True)
        data[data.columns[i]] = data[data.columns[i]].apply(lambda x: to_float_str(x))


    columns_to_impute = []
    for i, column in enumerate(data.columns):
        if i not in categorical_features and pd.isnull(data[column]).any():
            columns_to_impute.append(column)
    for column_name in columns_to_impute:
        data[column_name + "_imputed"] = pd.isnull(data[column_name]).astype(float)
        data[column_name].fillna(0, inplace=True)

    return data, targets, [data.columns[x] for x in categorical_features]