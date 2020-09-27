#!/home/vvsopov/anaconda3/bin/python

from base_experiment import BaseExperiment
from kdd_base import get_dataset


class UpsellingExperiment(BaseExperiment):
    POSITIVE_STEPS = [900, 1800, 3682]
    NEGATIVE_STEPS = [3000, 6000, 12000, 24000, 46318]
    PLOT_FIG_SIZE = (12, 16)

    def get_dataset(self):
        return get_dataset('upselling')
    
    
if __name__ == '__main__':  
    UpsellingExperiment().run()