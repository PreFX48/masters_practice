#!/home/vvsopov/anaconda3/bin/python

from base_experiment import BaseExperiment
from kdd_base import get_dataset


class AppetencyExperiment(BaseExperiment):
    POSITIVE_STEPS = [200, 450, 890]
    NEGATIVE_STEPS = [1500, 3000, 6000, 12000, 24000, 49110]
    PLOT_FIG_SIZE = (12, 16)

    def get_dataset(self):
        return get_dataset('appetency')
    
    
if __name__ == '__main__':  
    AppetencyExperiment().run()