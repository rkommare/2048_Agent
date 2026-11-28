from generate_runs import generate_batches
from policy import Policy
from world import Board
from cnn import CNN
import torch
from torch.nn import MSELoss
from torch.optim import Adam
from matplotlib import pyplot as plt
import time
import random

def dataset(lam,device='cpu',validation_split=.2):
    policy = Policy()
    f = open('rollouts.txt')
    total_length = 0
    total_score = 0
    test_states = []
    test_targets = []
    lines = f.readlines()
    for line in lines[:int(validation_split*len(lines))]:
        future_reward = 0
        length,score,*rollout = line.strip().split(',')
        rollout.reverse()
        total_length += int(length)
        total_score += int(score)
        while len(rollout) > 0:
            reward,action,state,*rollout = rollout
            future_reward = float(reward) + lam * future_reward
            board = Board(state)
            test_states += [[board.board]]
            target = [float('nan'),float('nan'),float('nan'),float('nan')]
            target[policy.action_to_index[action]] = future_reward
            test_targets += [target]
    train_states = []
    train_targets = []
    for line in lines[int(validation_split*len(lines)):]:
        future_reward = 0
        length,score,*rollout = line.strip().split(',')
        rollout.reverse()
        total_length += int(length)
        total_score += int(score)
        while len(rollout) > 0:
            reward,action,state,*rollout = rollout
            future_reward = float(reward) + lam * future_reward
            board = Board(state)
            train_states += [[board.board]]
            target = [float('nan'),float('nan'),float('nan'),float('nan')]
            target[policy.action_to_index[action]] = future_reward
            train_targets += [target]
    f.close()
    return torch.tensor(train_states,device=device,dtype=torch.float32),torch.tensor(train_targets,device=device,dtype=torch.float32),torch.tensor(test_states,device=device,dtype=torch.float32),torch.tensor(test_targets,device=device,dtype=torch.float32),total_length,total_score

def rotated_dataset(lam,device='cpu',validation_split=.2):
    policy = Policy()
    f = open('rollouts.txt')
    total_length = 0
    total_score = 0
    test_states = []
    test_targets = []
    lines = f.readlines()
    for line in lines[:int(validation_split*len(lines))]:
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
            test_states += [[board.board]]
            target = [float('nan'),float('nan'),float('nan'),float('nan')]
            target[policy.action_to_index[directions[policy.action_to_index[action]]]] = future_reward
            test_targets += [target]
    train_states = []
    train_targets = []
    for line in lines[int(validation_split*len(lines)):]:
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
            train_states += [[board.board]]
            target = [float('nan'),float('nan'),float('nan'),float('nan')]
            target[policy.action_to_index[directions[policy.action_to_index[action]]]] = future_reward
            train_targets += [target]
    f.close()
    return torch.tensor(train_states,device=device,dtype=torch.float32),torch.tensor(train_targets,device=device,dtype=torch.float32),torch.tensor(test_states,device=device,dtype=torch.float32),torch.tensor(test_targets,device=device,dtype=torch.float32),total_length,total_score

def loss_fn(output,target):
    mask = (torch.logical_not(torch.isnan(target)))
    return MSELoss()(output[mask],target[mask])

def train_cnn(epochs,subepoch,batch_size,num_batches,exploration_rate,lr,lam,channels):
    policy = Policy()
    policy.cnn = CNN(channels)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy.set_device(device)
    optimizer = Adam(policy.cnn.parameters(),lr=lr)
    print('Device:',device)
    avg_len = []
    avg_score = []
    avg_loss = []
    avg_counts = {'left':[],'right':[],'up':[],'down':[]}
    for epoch in range(epochs):
        print('Iteration:',epoch+1)

        if epoch % subepoch == 0:
            stats = generate_batches(batch_size,num_batches,policy,exploration_rate)
            counts = stats['counts']
            x_train,y_train,x_test,y_test,length,score = rotated_dataset(lam,device)

        policy.cnn.train()
        optimizer.zero_grad()

        preds = policy.cnn(x_train)
        loss = loss_fn(preds,y_train)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            policy.cnn.eval()
            preds = policy.cnn(x_test)
            loss = loss_fn(preds,y_test)

        avg_len += [length/batch_size]
        avg_score += [score/batch_size]
        avg_loss += [loss.item()]
        for key in avg_counts:
            avg_counts[key] += [counts[key]/length]
    policy.save_cnn('./container_files/cnn_weights.pth')
    return avg_len,avg_score,avg_loss,avg_counts

if __name__ == '__main__':
    start = time.time()
    avg_len,avg_score,avg_loss,avg_counts = train_cnn(2500,500,1000,1,0,.001,.5,2)
    print('Time:',(time.time()-start)/60,'Minutes')
    fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
    ax1.plot(avg_loss)
    ax1.set_title('average loss vs iteration')
    ax2.plot(avg_score)
    ax2.set_title('average score vs iteration')
    ax3.plot(avg_len)
    ax3.set_title('average length vs iteration')
    for key in avg_counts:
        ax4.plot(avg_counts[key],label=key)
    ax4.legend()
    ax4.set_title('directions frequency vs iteration')
    plt.tight_layout()
    plt.savefig('./container_files/cnn_plot.png')
    plt.show()