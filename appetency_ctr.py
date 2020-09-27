#!/home/vvsopov/anaconda3/bin/python

from appetency  import AppetencyExperiment


class AppetencyExperimentCtr(AppetencyExperiment):
    USE_CTR = True
    
    
if __name__ == '__main__':  
    AppetencyExperimentCtr().run()