import random

class Policy():
    def __init__(self):
        self.table = None
        self.cnn = None
        self.index_to_action = {0:'left',1:'right',2:'up',3:'down'}
        self.action_to_index = {'left':0,'right':1,'up':2,'down':3}

    def load_policy(self,filename):
        pass

    def load_cnn(self,filename):
        pass

    def expected_values(self,state):
        if self.table is None and self.cnn is None:
            values = [('left',0),('right',0),('up',0),('down',0)]
        random.shuffle(values)
        values.sort(key=lambda x:x[1],reverse=True)
        return values