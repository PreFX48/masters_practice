#!/home/vvsopov/anaconda3/bin/python

from upselling  import UpsellingExperiment


class UpsellingExperimentCtr(UpsellingExperiment):
    USE_CTR = True
    
    
if __name__ == '__main__':  
    UpsellingExperimentCtr().run()