import numpy as np
from catboost import datasets
import pandas as pd
from tensorflow import keras

from base_experiment import BaseExperiment

class AmazonExperiment(BaseExperiment):
    POSITIVE_STEPS = [2000, 5000, 10000, 20000, 30872]
    NEGATIVE_STEPS = [500, 1000, 1897]
    PLOT_FIG_SIZE = (8, 10.67)

    def get_dataset(self):
        train_df, _ = datasets.amazon()
#         train_df = train_df[:500]
        y = train_df['ACTION']
        X = train_df.drop(columns='ACTION') # or X = train_df.drop('ACTION', axis=1)
        return X, y, X.columns

if __name__ == '__main__':  
    AmazonExperiment().run()