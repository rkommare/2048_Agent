from generate_runs import generate
from policy import Policy
from world import Board
from cnn import CNN
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import MSELoss

def dataset(lam):
    policy = Policy()
    f = open('rollouts.txt')
    total_length = 0
    total_score = 0
    states = []
    targets = []
    for line in f.readlines():
        future_reward = 0
        length,score,*rollout = line.strip().split(',')
        rollout.reverse()
        total_length += int(length)
        total_score += int(score)
        while len(rollout) > 0:
            reward,action,state,*rollout = rollout
            future_reward = float(reward) + lam * future_reward
            target = [0,0,0,0]
            target[policy.action_to_index[action]] = future_reward
            board = Board(state)
            states += [board.board]
            targets += [target]
    f.close()
    return torch.tensor(states).to(torch.float32),torch.tensor(targets).to(torch.float32)

def loss_fn(output,target):
    '''for i,y in enumerate(target):
        for j,yj in enumerate(y):
            if yj == 0:
                output[i][j] = 0
    print(output)
    return fn(output,target)'''
    mask = (target != 0)
    return MSELoss()(output[mask],target[mask])

'''generate(1,Policy(),.4)
states,targets = dataset(.9)
states = states[:1]
targets = targets[:1]
cnn = CNN()
print(len(states))
print(states.shape)
output = cnn(states)
print(output)
print(targets)
loss = loss_fn(output,targets)
loss.backward()
print(cnn.temp.weight.grad)'''

policy = Policy()
policy.cnn = CNN()
generate(1,policy,.4)
    
