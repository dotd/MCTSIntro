
import numpy as np
#import types

class FourInARow():
    '''
    There are two players: -1 and 1. 0 is empty slot.

    '''

    def __init__(self, **kwargs):
        '''
        :param height: of the board
        :param width: of the board
        player 1 always starts
        '''
        self.height = kwargs.get("height", 6)
        self.width = kwargs.get("width", 7)
        self.board = kwargs.get("board", np.zeros((self.height, self.width)))
        self.N_win = kwargs.get("N_win", 4)
        self.is_winner_flag = kwargs.get("is_winner_flag", False)
        self.turn = kwargs.get("turn", 1)

    def get_possible_actions(self):
        '''
        Returns all the possible actions.
        :return: a map of (col -> what row is free?)
        '''
        res = {}
        for col in range(self.width):
            for row in range(self.height):
                if self.board[row][col]==0:
                    res[col] = row
                    break
        return res

    def get_state_copy(self):
        return np.copy(self.board), np.copy(self.turn)

    def set_state_as_copy(self, board, turn):
        self.board = np.copy(board)
        self.turn = np.copy(turn)

    def play(self, player_num, column_idx):
        '''
        plays the move and if the move is valid, updates the state
        :param player_num: which player is playing?
        :param column_idx: what is the action the player wants to do?
        :return: (boolean, boolean) = (if the move is possible, was that a win?)
        '''
        possible_action = self.get_possible_actions()
        if column_idx in possible_action.keys():
            free_row = possible_action[column_idx]
            self.board[free_row][column_idx] = player_num
            self.turn = -self.turn
            self.is_winner_flag = self.is_winner(free_row, column_idx)
            return (True, self.is_winner_flag)
        return (False, (False, None, None))

    def is_winner(self,row, col):
        player = self.board[row][col]
        board = self.board==player
        # straight down check
        if row>=self.N_win-1:
            idx_start = row-self.N_win+1
            idx_end = row
            vec_y,vec_x = list(range(idx_start,idx_end+1)),[col] * self.N_win
            vec = board[vec_y, vec_x]
            if vec.all():
                return (True,vec_y,vec_x)

        for k in range(-self.N_win+1,0):
            # horizontal check
            x_start = col+k
            if x_start<0:
                continue
            x_end = x_start + self.N_win - 1
            if x_end>= self.width:
                continue
            vec_y, vec_x = row, range(x_start,x_end+1)
            vec = board[vec_y, vec_x]
            if vec.all():
                return (True,vec_y,vec_x)

            # Bottom-left to upper-right
            y_low_start = row + k
            if y_low_start<0:
                continue
            y_low_end = y_low_start + self.N_win - 1
            if y_low_end>=self.height:
                continue
            vec_y, vec_x = range(y_low_start,y_low_end+1), range(x_start,x_end+1)
            vec = board[vec_y,vec_x]
            if vec.all():
                return (True,vec_y,vec_x)

            y_high_start = row - k
            if y_high_start >= self.height:
                continue
            y_high_end = y_high_start - (self.N_win - 1)
            if y_high_end<0:
                continue
            vec_y, vec_x = list(range(y_high_start, y_high_end-1, -1)), list(range(x_start, x_end+1))
            vec = board[vec_y,vec_x]
            if vec.all():
                return (True,vec_y,vec_x)

        return (False, None, None)

    def show_board(self):
        lines = []
        for row in range(self.height-1,-1,-1):
            line = []
            for col in range(self.width):
                # creating the lines by joining the +-0 signs
                line.append("+" if self.board[row][col]>0 else ("o" if self.board[row][col]==0 else "-"))
            # appending the lines
            lines.append("".join(line))
        # joining the lines
        return "\n".join(lines)

def show_random_game():
    four = FourInARow()
    print("init board:")
    print(four.show_board())
    for p in range(four.height*four.width):
        player = four.turn
        print("turn is of player ", player)
        possible_actions = four.get_possible_actions()
        if bool(possible_actions)==False:
            print("No possible moves. Exit")
            break
        print("possible columns/actions ", possible_actions)
        # choose a random action
        a = np.random.choice(list(possible_actions.keys()))
        print("the action is ", a)
        success, is_winner = four.play(player,a)
        print("success=" + str(success))
        print("is_winner=" + str(is_winner))
        print("board in the turn end:")
        print(four.show_board())
        if is_winner[0]:
            print("player " + str(player) + " is a winner! exit.")
            break

show_random_game()


