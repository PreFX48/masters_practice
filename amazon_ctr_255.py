#!/home/vvsopov/anaconda3/bin/python

from amazon import AmazonExperiment


class AmazonExperimentCtr255(AmazonExperiment):
    USE_CTR = True
    KERAS_MAX_ONEHOT_VALUES = 255


if __name__ == '__main__':  
    AmazonExperimentCtr255().run()