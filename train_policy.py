from policy import Policy
from generate_runs import generate
from world import Board
from matplotlib import pyplot as plt
import os
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import time

def multiprocess(line,policy,a,lam,lock):
    unseen_states = 0
    future_reward = 0
    length,score,*rollout = line.strip().split(',')
    rollout.reverse()
    steps = []
    while len(rollout) > 0:
        reward,action,state,*rollout = rollout
        future_reward = float(reward) + lam * future_reward
        rotations = policy.rotate_state(state)
        steps += [(state,rotations,future_reward,action)]
        '''
        with lock:
            found = policy.find(rotations)
            if found is None:
                policy.table[state] = [0,0,0,0]
                unseen_states += 1
            else:
                state,_,directions = found
                action = directions[policy.action_to_index[action]]
            policy.table[state][policy.action_to_index[action]] = (1-a) * policy.table[state][policy.action_to_index[action]] + a * future_reward
        '''
    return steps,int(length),int(score),unseen_states

def train_mc(weight_iterations,n_per_iteration,exploration_rate,a,lam):
    policy = Policy()
    log = []
    cores = os.cpu_count()
    m = multiprocessing.Manager()
    lock = m.Lock()
    policy.table = m.dict()
    for i in range(weight_iterations):
        print('Iteration', i+1)
        generate(n_per_iteration,policy,exploration_rate)
        
        total_length = 0
        total_score = 0
        unseen_states = 0

        f = open('rollouts.txt')
        lines = f.readlines()

        with ProcessPoolExecutor(cores) as executor:
            try:
                multiprocesses = executor.map(multiprocess,lines,[policy] * n_per_iteration,[a] * n_per_iteration,[lam] * n_per_iteration,[lock] * n_per_iteration)
            except ValueError:
                print(("Error multiprocesses"))
        for steps,length,score,us in multiprocesses:
            total_length += length
            total_score += score
            unseen_states += us
            for state,rotations,future_reward,action in steps:
                found = policy.find(rotations)
                if found is None:
                    policy.table[state] = [0,0,0,0]
                    unseen_states += 1
                else:
                    state,_,directions = found
                    action = directions[policy.action_to_index[action]]
                policy.table[state][policy.action_to_index[action]] = (1-a) * policy.table[state][policy.action_to_index[action]] + a * future_reward

        '''
        for line in f:
            future_reward = 0
            length,score,*rollout = line.strip().split(',')
            rollout.reverse()
            while len(rollout) > 0:
                reward,action,state,*rollout = rollout
                found = policy.find(state)
                if found is None:
                    policy.table[state] = [0,0,0,0]
                    unseen_states += 1
                else:
                    state,_,directions = found
                    action = directions[policy.action_to_index[action]]
                future_reward = float(reward) + lam * future_reward
                policy.table[state][policy.action_to_index[action]] = (1-a) * policy.table[state][policy.action_to_index[action]] + a * future_reward
            total_length += int(length)
            total_score += int(score)
        '''
        f.close()
        log += [{'avg_len':total_length/n_per_iteration,'avg_score':total_score/n_per_iteration,'percent_unseen':unseen_states/total_length,'table_size:':len(policy.table)}]
    return log

if __name__ == '__main__':
    start = time.time()
    log = train_mc(500,100,.4,.3,.9)
    scores = []
    unseen = []
    for l in log:
        print(l)
        scores.append(l['avg_score'])
        unseen.append(l['percent_unseen'])
    print('Time:',(time.time()-start)/60)
    fig,(ax1,ax2) = plt.subplots(1,2)
    ax1.plot(scores)
    ax1.set_title('average score vs iteration')
    ax2.plot(unseen)
    ax2.set_title('percent unseen vs iteration')
    plt.savefig('plot.png')
    plt.show()
