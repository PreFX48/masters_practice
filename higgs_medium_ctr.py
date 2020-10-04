#!/home/vvsopov/anaconda3/bin/python

from higgs_medium import HiggsMediumExperiment


class HiggsMediumExperimentCtr(HiggsMediumExperiment):
    USE_CTR = True


if __name__ == '__main__':  
    HiggsMediumExperimentCtr().run()