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
                if column == 0:
                    string += '    |'
                else :
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

    def spawn(self):
        r,c = random.choice(self.get_empty())
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

    def __eq__(self,other):
        for row in range(len(self.board)):
            for column in range(len(self.board)):
                if self.board[row][column] != other.board[row][column]:
                    return False
        return True

    def copy(self):
        copy = Board()
        for row in range(len(self.board)):
            for column in range(len(self.board)):
                copy.board[row][column] = self.board[row][column]
        copy.score = self.score
        return copy
    
    def legal(self):
        legal_moves = []

        copy = self.copy()
        copy.move_left()
        if self != copy:
            legal_moves += ['left']

        copy = self.copy()
        copy.move_right()
        if self != copy:
            legal_moves += ['right']

        copy = self.copy()
        copy.move_up()
        if self != copy:
            legal_moves += ['up']

        copy = self.copy()
        copy.move_down()
        if self != copy:
            legal_moves += ['down']   

        return legal_moves     

if __name__ == '__main__':
    board = Board()
    while True:
        board.spawn()
        board.display_board()
        legal_moves = board.legal()
        if len(legal_moves) == 0:
            print('Game Over')
            break
        valid = False
        while not valid:
            x = input()
            if (x == 'a' or x == 'A') and 'left' in legal_moves:  
                board.move_left()
                valid = True
            elif (x == 'd' or x == 'D') and 'right' in legal_moves:
                board.move_right()
                valid = True
            elif (x == 'w' or x == 'W') and 'up' in legal_moves:
                board.move_up()
                valid = True
            elif (x == 's' or x == 'S') and 'down' in legal_moves:
                board.move_down()
                valid = True
        print('Score: %d' % board.score)