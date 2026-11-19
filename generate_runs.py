import random
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

if __name__ == '__main__':
    generate(3)