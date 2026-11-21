import random
import time
from world import Board
from policy import Policy
import os
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

def multiprocess(policy,exploration_rate,lock):
    board = Board()
    board.spawn()
    mapping = {'left':board.move_left,'right':board.move_right,'up':board.move_up,'down':board.move_down}
    length = 1
    score = 0
    exploited = 0
    explored = 0
    counts = {'left':0,'right':0,'up':0,'down':0}
    record = ''
    legal = board.legal()
    while len(legal) > 0:
        state = str(board)
        expected_values = policy.expected_values(state)
        exploit,*explore = [ev[0] for ev in expected_values if ev[0] in legal]
        if len(explore) > 0 and random.random() < exploration_rate:
            move = random.choice(explore)
            explored += 1
        else:
            move = exploit
            exploited += 1
        counts[move] += 1
        mapping[move]()
        reward = board.score - score
        record += ',' + state + ',' + move + ',' + str(reward)
        score = board.score
        length += 1
        board.spawn()
        legal = board.legal()
    record = str(length) + ',' + str(score) + record + '\n'
    with lock:
        f = open('rollouts.txt','a')
        f.write(record)
        f.close()
    return exploited,explored,counts

def generate(n,policy,exploration_rate):
    f = open('rollouts.txt','w')
    f.close()
    '''
    exploited = 0
    explored = 0
    counts = {'left':0,'right':0,'up':0,'down':0}
    for i in range(n):
        board = Board()
        board.spawn()
        mapping = {'left':board.move_left,'right':board.move_right,'up':board.move_up,'down':board.move_down}
        length = 1
        score = 0
        record = ''
        legal = board.legal()
        while len(legal) > 0:
            state = str(board)
            expected_values = policy.expected_values(state)
            exploit,*explore = [ev[0] for ev in expected_values if ev[0] in legal]
            if len(explore) > 0 and random.random() < exploration_rate:
                move = random.choice(explore)
                explored += 1
            else:
                move = exploit
                exploited += 1
            counts[move] += 1
            mapping[move]()
            reward = board.score - score
            record += ',' + state + ',' + move + ',' + str(reward)
            score = board.score
            length += 1
            board.spawn()
            legal = board.legal()
        record = str(length) + ',' + str(score) + record + '\n'
        f.write(record)
    f.close()
    '''
    cores = os.cpu_count()
    m = multiprocessing.Manager()
    lock = m.Lock()
    with ProcessPoolExecutor(cores) as executor:
        try:
            multiprocesses = executor.map(multiprocess,[policy] * n,[exploration_rate] * n,[lock] * n)
        except ValueError:
            print(("Error multiprocesses"))
    total_exploited = 0
    total_explored = 0
    total_counts = {'left':0,'right':0,'up':0,'down':0}
    for exploited,explored,counts in multiprocesses:
        total_exploited += exploited
        total_explored += explored
        for key in counts:
            total_counts[key] += counts[key]
    return {'exploited':total_exploited,'explored':total_explored,'counts':total_counts}

def play(index):
    f = open('rollouts.txt')
    record = f.readlines()[index].strip().split(',')
    length, score, *rollout = record
    print('Length:',length)
    print('Final Score:',score)
    score = 0
    while len(rollout) > 0:
        time.sleep(.5)
        state,action,reward,*rollout = rollout
        board = Board(state)
        board.display_board()
        print(action)
        score += int(reward)
        print('Score:',score)


if __name__ == '__main__':
    print(generate(100,Policy(),.5))
    play(0)