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

    def move_left(self):
        for row in range(len(self.board)):
            free = 0
            i = 0
            combined = [0] * 4
            while i < len(self.board):
                if self.board[row][i] == 0:
                    i += 1
                else:
                    if i > free:
                        self.board[row][free] = self.board[row][i]
                        self.board[row][i] = 0
                    if free > 0 and self.board[row][free] == self.board[row][free-1] and combined[free-1] == 0:
                            self.board[row][free-1] *= 2
                            self.board[row][free] = 0
                            combined[free-1] = 1
                            self.score += self.board[row][free-1]
                            i += 1
                    else:
                        free += 1
                        i+= 1

    def flip_horizontal(self):
        for row in self.board:
            row.reverse()

    def move_right(self):
        self.flip_horizontal()
        self.move_left()
        self.flip_horizontal()

    def transpose(self):
        for row in range(len(self.board)):
            k = 0
            while row + k < len(self.board):
                temp = self.board[row][row+k]
                self.board[row][row+k] = self.board[row+k][row]
                self.board[row+k][row] = temp
                k += 1

    def move_up(self):
        self.transpose()
        self.move_left()
        self.transpose()

    def move_down(self):
        self.transpose()
        self.move_right()
        self.transpose()

if __name__ == '__main__':
    board = Board()
    board.display_board()
    while len(board.get_empty()) > 0:
        empty = board.get_empty()
        x = input()
        if x == 'a' or x == 'A':  
            board.move_left()
        elif x == 'd' or x == 'D':
            board.move_right()
        elif x == 'w' or x == 'W':
            board.move_up()
        elif x == 's' or x == 'S':
            board.move_down()
        else:
            continue
        board.spawn(empty)
        board.display_board()