#!/home/vvsopov/anaconda3/bin/python

from amazon import AmazonExperiment


class AmazonExperimentCtr(AmazonExperiment):
    USE_CTR = True


if __name__ == '__main__':  
    AmazonExperimentCtr().run()