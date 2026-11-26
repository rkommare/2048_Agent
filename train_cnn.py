from generate_runs import generate
from policy import Policy
from world import Board
from cnn import CNN
import torch
from torch.nn import MSELoss
import time
import random

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
            board = Board(state)
            states += [board.board]
            target = [0,0,0,0]
            target[policy.action_to_index[action]] = future_reward
            targets += [target]
    f.close()
    return torch.tensor(states).to(torch.float32),torch.tensor(targets).to(torch.float32)

def rotated_dataset(lam):
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
            rotations = policy.rotate_state(state)
            state,directions = random.choice(rotations)
            board = Board(state)
            states += [board.board]
            target = [0,0,0,0]
            target[policy.action_to_index[directions[policy.action_to_index[action]]]] = future_reward
            targets += [target]
    f.close()
    return torch.tensor(states).to(torch.float32),torch.tensor(targets).to(torch.float32)

def loss_fn(output,target):
    mask = (target != 0)
    return MSELoss()(output[mask],target[mask])

if __name__ == '__main__':
    policy = Policy()
    policy.cnn = CNN()
    start = time.time()
    generate(500,policy,.4)
    print('Time:',time.time()-start)