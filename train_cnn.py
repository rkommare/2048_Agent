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

def dataset(lam,device='cpu'):
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
            states += [[board.board]]
            target = [float('nan'),float('nan'),float('nan'),float('nan')]
            target[policy.action_to_index[action]] = future_reward
            targets += [target]
    f.close()
    return torch.tensor(states,device=device,dtype=torch.float32),torch.tensor(targets,device=device,dtype=torch.float32),total_length,total_score

def rotated_dataset(lam,device='cpu'):
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
            states += [[board.board]]
            target = [float('nan'),float('nan'),float('nan'),float('nan')]
            target[policy.action_to_index[directions[policy.action_to_index[action]]]] = future_reward
            targets += [target]
    f.close()
    return torch.tensor(states,device=device,dtype=torch.float32),torch.tensor(targets,device=device,dtype=torch.float32),total_length,total_score

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
    for epoch in range(epochs):
        print('Iteration:',epoch+1)

        if epoch % subepoch == 0:
            generate_batches(batch_size,num_batches,policy,exploration_rate)
            states,targets,length,score = dataset(lam,device)

        policy.cnn.train()
        optimizer.zero_grad()

        preds = policy.cnn(states)
        loss = loss_fn(preds,targets)

        loss.backward()
        optimizer.step()

        avg_len += [length/batch_size]
        avg_score += [score/batch_size]
        avg_loss += [loss.item()]
    policy.save_cnn('./container_files/cnn_weights.pth')
    return avg_len,avg_score,avg_loss

if __name__ == '__main__':
    start = time.time()
    avg_len,avg_score,avg_loss = train_cnn(100,10,500,1,.2,.05,.9,4)
    print('Time:',(time.time()-start)/60,'Minutes')
    fig,(ax1,ax2,ax3) = plt.subplots(3,1)
    ax1.plot(avg_loss)
    ax1.set_title('average loss vs iteration')
    ax2.plot(avg_score)
    ax2.set_title('average score vs iteration')
    ax3.plot(avg_len)
    ax3.set_title('average length vs iteration')
    plt.tight_layout()
    plt.savefig('./container_files/cnn_plot.png')
    plt.show()