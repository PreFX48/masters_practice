#!/home/vvsopov/anaconda3/bin/python

from amazon import AmazonExperiment


class AmazonExperimentCtr2(AmazonExperiment):
    USE_CTR = True
    KERAS_MAX_ONEHOT_VALUES = 2


if __name__ == '__main__':  
    AmazonExperimentCtr2().run()