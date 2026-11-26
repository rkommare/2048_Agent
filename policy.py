import random
import pickle
import torch
from world import Board
from cnn import CNN

class Policy():
    def __init__(self):
        self.table = None
        self.cnn = None
        self.index_to_action = {0:'left',1:'right',2:'up',3:'down'}
        self.action_to_index = {'left':0,'right':1,'up':2,'down':3}

    def save_policy(self,filename):
        with open(filename,'wb') as f:
            pickle.dump(self.table,f)
    
    def load_policy(self,filename):
        with open(filename,'rb') as f:
            self.table = pickle.load(f)

    def save_cnn(self,filename):
        torch.save(self.cnn.state_dict(),filename)

    def load_cnn(self,filename):
        self.cnn = CNN()
        self.cnn.load_state_dict(torch.load(filename, weights_only=True))

    def rotate_state(self,state):
        rotations = []
        rotate_map = {'left':'down','down':'right','right':'up','up':'left'}
        board = Board(state)
        directions = ['left','right','up','down']
        for i in range(4):
            board.rotate()
            directions = list(map(lambda x:rotate_map[x],directions))
            rotations.append((str(board),directions))
        board = Board(state)
        board.flip_horizontal()
        directions = ['right','left','up','down']
        for i in range(4):
            board.rotate()
            directions = list(map(lambda x:rotate_map[x],directions))
            rotations.append((str(board),directions))
        return rotations

    def find(self,rotations):
        for state,directions in rotations:
            if state in self.table:
                return state,self.table[state],directions
        return None

    def expected_values(self,state):
        if self.table is None and self.cnn is None:
            values = [('left',0),('right',0),('up',0),('down',0)]
        elif self.cnn is None:
            rotations = self.rotate_state(state)
            found = self.find(rotations)
            if found is not None:
                _, evs, directions = found
                values = [(self.index_to_action[i],evs[self.action_to_index[d]])for i,d in enumerate(directions)]
            else:
                values = [('left',0),('right',0),('up',0),('down',0)]
        else:
            self.cnn.eval()
            board = Board(state)
            tensor = torch.tensor([board.board]).to(torch.float32)
            evs = self.cnn(tensor)[0]
            values = [(self.index_to_action[i],ev.item()) for i,ev in enumerate(evs)]
        random.shuffle(values)
        values.sort(key=lambda x:x[1],reverse=True)
        return values