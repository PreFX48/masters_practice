#!/home/vvsopov/anaconda3/bin/python

from higgs import HiggsSmallExperiment


class HiggsSmallExperimentCtr(HiggsSmallExperiment):
    USE_CTR = True


if __name__ == '__main__':  
    HiggsSmallExperimentCtr().run()