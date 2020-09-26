from kdd import AppetencyExperiment, UpsellingExperiment


class AppetencyExperimentCtr(AppetencyExperiment):
    USE_CTR = True


class UpsellingExperimentCtr(UpsellingExperiment):
    USE_CTR = True
    
    
if __name__ == '__main__':  
    AppetencyExperimentCtr().run()
    UpsellingExperimentCtr().run()