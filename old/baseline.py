'''
Class Baseline
'''

class Baseline:
    def __init__(self,name, type_of_wordvec):
        self.baseline_name = name
        self.type_of_wordvec = type_of_wordvec
        # self.print_information()
    
    def train(self, x_train, y_train):
        raise NotImplementedError

    def save_model(self, save_path):
        raise NotImplementedError

    def predict(self, x_test):
        raise NotImplementedError

    def print_information(self):
        print('=========================')
        print('{} baseline',self.baseline_name)
        print('WordVec model: {}',self.type_of_wordvec)