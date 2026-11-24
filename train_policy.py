from policy import Policy
from generate_runs import generate
from matplotlib import pyplot as plt
import time

def train_mc(weight_iterations,n_per_iteration,exploration_rate,a,lam):
    policy = Policy()
    log = []
    policy.table = {}
    for i in range(weight_iterations):
        print('Iteration', i+1)
        generate(n_per_iteration,policy,exploration_rate)
        total_length = 0
        total_score = 0
        unseen_states = 0

        f = open('rollouts.txt')
        for line in f.readlines():
            future_reward = 0
            length,score,*rollout = line.strip().split(',')
            rollout.reverse()
            total_length += int(length)
            total_score += int(score)
            while len(rollout) > 0:
                reward,action,state,*rollout = rollout
                future_reward = float(reward) + lam * future_reward
                rotations = Policy().rotate_state(state)
                found = policy.find(rotations)
                if found is None:
                    policy.table[state] = [0,0,0,0]
                    unseen_states += 1
                else:
                    state,_,directions = found
                    action = directions[policy.action_to_index[action]]
                policy.table[state][policy.action_to_index[action]] = (1-a) * policy.table[state][policy.action_to_index[action]] + a * future_reward
        log += [{'avg_len':total_length/n_per_iteration,'avg_score':total_score/n_per_iteration,'percent_unseen':unseen_states/total_length,'table_size:':len(policy.table)}]
        f.close()
    policy.save_policy('./container_files/mc_policy.pkl')
    return log

if __name__ == '__main__':
    start = time.time()
    log = train_mc(100,100,.4,.3,.9)
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
    plt.savefig('./container_files/plot.png')
    plt.show()
