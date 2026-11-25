from generate_runs import generate
from policy import Policy
from world import Board
import pandas
import time
import random

def dataset(lam):
    f = open('rollouts.txt')
    total_length = 0
    total_score = 0
    data = []
    for line in f.readlines():
        future_reward = 0
        length,score,*rollout = line.strip().split(',')
        rollout.reverse()
        total_length += int(length)
        total_score += int(score)
        while len(rollout) > 0:
            reward,action,state,*rollout = rollout
            future_reward = float(reward) + lam * future_reward
            data += [[state,action,future_reward]]
            #board = Board(state)
            #data += [[board.board,action,future_reward]]
    df = pandas.DataFrame(data,columns=['state','move','reward'])
    f.close()
    return df

def rotated_dataset(lam):
    policy = Policy()
    f = open('rollouts.txt')
    total_length = 0
    total_score = 0
    data = []
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
            data += [[state,directions[policy.action_to_index[action]],future_reward]]
            #board = Board(state)
            #data += [[board.board,directions[policy.action_to_index[action]],future_reward]]
    df = pandas.DataFrame(data,columns=['state','move','reward'])
    f.close()
    return df

if __name__ == '__main__':
    generate(500,Policy(),.4)
    start = time.time()
    dataset(.9)
    print('Time:',time.time()-start)
    start = time.time()
    rotated_dataset(.9)
    print('Time:',time.time()-start)