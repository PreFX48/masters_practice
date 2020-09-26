from adult import AdultExperiment


class AdultExperimentCtr(AdultExperiment):
    USE_CTR = True

    
if __name__ == '__main__':  
    AdultExperimentCtr().run()