import random

class Board:
    def __init__(self):
        self.board = [[0] * 4 for _ in range(4)]
        self.score = 0

    def display_board(self):
        line = ''.join(['-'] * 21)
        print(line)
        for row in self.board:
            string = '|'
            for column in row:
                string += "{num:4d}".format(num=column) + '|'
            print(string)
            print(line)

    def get_empty(self):
        empty = []
        for row in range(len(self.board)):
            for column in range(len(self.board)):
                if self.board[row][column] == 0:
                    empty.append((row,column))
        return empty

    def spawn(self,empty):
        r,c = random.choice(empty)
        self.board[r][c] = random.choice([2] * 9 + [4])

if __name__ == '__main__':
    board = Board()
    while len(board.get_empty()) > 0:
        board.display_board()
        empty = board.get_empty()
        x = input()
        board.spawn(empty)
    board.display_board()