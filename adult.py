import numpy as np
import pandas as pd
from tensorflow import keras

from base_experiment import BaseExperiment

class AdultExperiment(BaseExperiment):
    POSITIVE_STEPS = [500, 1250, 2500, 5000, 7841]
    NEGATIVE_STEPS = [2500, 5000, 10000, 20000, 24720]

    def get_dataset(self):
        df = pd.read_csv('adult/adult.data')
#         df = df[:500]
        y = df.label.apply(lambda x: 0 if x.strip() == '<=50K' else 1)
        X = df.drop(columns=['label'])
        cat_features = [
            'workclass', 'education', 'marital_status', 'occupation', 'relationship',
            'race', 'sex', 'native_country',
        ]
        return X, y, cat_features
    
if __name__ == '__main__':  
    AdultExperiment().run()