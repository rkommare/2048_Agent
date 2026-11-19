import random
import time
from world import Board

def generate(n,policy=None):
    f = open('rollouts.txt','w')
    if policy is not None:
        pass
    else:
        for i in range(n):
            board = Board()
            board.spawn()
            mapping = {'left':board.move_left,'right':board.move_right,'up':board.move_up,'down':board.move_down}
            length = 1
            score = 0
            record = ''
            while len(board.legal()) > 0:
                move = random.choice(board.legal())
                mapping[move]()
                reward = board.score - score
                record += ',' + str(board) + ',' + move + ',' + str(reward)
                score = board.score
                length += 1
                board.spawn()
            record = str(length) + ',' + str(score) + record + '\n'
            f.write(record)
    f.close()

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
        print(action)
        board = Board(state)
        board.display_board()
        score += int(reward)
        print('Score:',score)


if __name__ == '__main__':
    generate(3)
    play(0)