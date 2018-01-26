from FourInARow import FourInARow
import datetime
from random import choice
import numpy as np

class MonteCarlo(object):
    def __init__(self, board, **kwargs):
        self.four_in_a_row = FourInARow()
        self.states = []
        seconds = kwargs.get('time', 30)
        self.calculation_time = datetime.timedelta(seconds=seconds)
        self.max_moves = kwargs.get('max_moves', 100)

    def update(self, state):
        self.states.append(state)

    def get_play(self):
        begin = datetime.datetime.utcnow()
        while datetime.datetime.utcnow() - begin < self.calculation_time:
            self.run_simulation()

    def run_simulation(self):
        '''
        We run the simulation on the copy.
        :return:
        '''
        board_copy, turn_copy = self.four_in_a_row.get_state_copy()
        four_in_a_row_copy = FourInARow(board=board_copy, turn=turn_copy)

        for t in range(self.max_moves):
            legal_actions = four_in_a_row_copy.get_possible_actions()

            action = choice(legal_actions)
            success, is_winner_vec = four_in_a_row_copy.play(turn_copy, action)

            if winner:
                break