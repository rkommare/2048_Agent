from generate_runs import generate
from policy import Policy
from world import Board
import pandas
import time
import random

def dataset(lam):
    f = open('rollouts.txt')
    df = pandas.DataFrame(columns=['state','move','reward'])
    total_length = 0
    total_score = 0
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
            df = pandas.concat([df,pandas.DataFrame([[board.board,action,future_reward]],columns=df.columns)],ignore_index=True)
    return df

def rotated_dataset(lam):
    policy = Policy()
    f = open('rollouts.txt')
    df = pandas.DataFrame(columns=['state','move','reward'])
    total_length = 0
    total_score = 0
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
            df = pandas.concat([df,pandas.DataFrame([[board.board,directions[policy.action_to_index[action]],future_reward]],columns=df.columns)],ignore_index=True)
    return df

if __name__ == '__main__':
    start = time.time()
    generate(3,Policy(),.4)
    #df = dataset(.9)
    df = rotated_dataset(.9)
    for _,row in df.head().iterrows():
        board = Board(row['state'])
        print(row['move'])
        board.display_board()
    for _,row in df.tail().iterrows():
        board = Board(row['state'])
        print(row['move'])
        board.display_board()
    print('Time:',time.time()-start)
