#!/home/vvsopov/anaconda3/bin/python

import numpy as np
import pandas as pd
from tensorflow import keras

from base_experiment import BaseExperiment

class HiggsSmallExperiment(BaseExperiment):
    POSITIVE_STEPS = [12000, 25000, 50000, 105676]
    NEGATIVE_STEPS = [12000, 25000, 50000, 94324]
    PLOT_FIG_SIZE = (8, 10.67)

    def get_dataset(self):
        columns = [
            'is_signal',
            'lepton_pT', 'lepton_eta', 'lepton_phi', 'missing_energy_magnitude', 'missing_energy_phi',
            'jet_1_pt', 'jet_1_eta', 'jet_1_phi', 'jet_1_b_tag',
            'jet_2_pt', 'jet_2_eta', 'jet_2_phi', 'jet_2_b_tag',
            'jet_3_pt', 'jet_3_eta', 'jet_3_phi', 'jet_3_b_tag',
            'jet_4_pt', 'jet_4_eta', 'jet_4_phi', 'jet_4_b_tag',
            'm_jj', 'm_jjj', 'm_lv', 'm_jlv', 'm_bb', 'm_wbb', 'm_wwbb',
        ]
        df = pd.read_csv('higgs_small.csv', names=columns)
        # df = df[:100]
        y = df['is_signal']
        X = df.drop(columns='is_signal')
        return X, y, []

if __name__ == '__main__':  
    HiggsSmallExperiment().run()