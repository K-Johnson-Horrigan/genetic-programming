import numpy as np
from src.utils.utils import to_tuple, cartesian_prod


class TM:
    """
    Class representing a multidimensional Turing Machine during execution.
    """

    ANY = -2 # Reserved value replaced with both a 0 and 1
    WALL = -1 # Reserved value that the machine cannot change on the tape

    HALT_STATE = -1

    def __init__(self, trans, tape=None, state=0):
        self.state = state
        self.state_history = [self.state]
        self.tape = tape or {}

        # Transition array, Single symbol head
        if type(trans) == np.ndarray:
            self.head_shape = None
            self.trans = trans
            self.N = len(trans[0,0]) - 2 # Length of transition from (0,0) minus the values for state and symbol
            self.head_pos = (0,) * self.N
            self.head_pos_history = [self.head_pos]

        # Transition dict, Single symbol head
        elif type(trans[0][1]) not in (tuple, list, np.ndarray):
            self.head_shape = None
            self.trans = {(t[0],t[1]): tuple(t[2:]) for t in trans}
            self.N = len(trans[0][4])
            self.head_pos = (0,) * self.N
            self.head_pos_history = [self.head_pos]

        # Transition dict, Multi symbol head
        # Most of this code is for parsing the wildcard symbols: ANY and WALL
        else:
            self.trans = {}
            self.head_shape = np.array(trans[0][1]).shape
            self.N = len(trans[0][4])
            self.head_pos = (0,) * self.N
            self.head_pos_history = [self.head_pos]
            for transition in trans:
                symbol_blocks = np.array(transition[1])
                symbol_block_shape = symbol_blocks.shape
                symbol_blocks = symbol_blocks.ravel()
                symbol_blocks = [[i] if i != TM.ANY else [TM.WALL, 0, 1] for i in symbol_blocks]
                symbol_blocks = cartesian_prod(*symbol_blocks)
                symbol_blocks = [i.reshape(symbol_block_shape) for i in symbol_blocks]
                for symbol_block in symbol_blocks:
                    new_symbol_block = np.array(transition[3])
                    # TM.ANY is replaced with the value in replaced with the value in the original symbol block
                    new_symbol_block[new_symbol_block == TM.ANY] = symbol_block[new_symbol_block == TM.ANY]
                    new_symbol_block[symbol_block == TM.WALL] = TM.WALL
                    new_symbol_block = to_tuple(new_symbol_block)
                    symbol_blocks = to_tuple(symbol_block)
                    self.trans[(transition[0], symbol_blocks)] = (transition[2], new_symbol_block, *transition[4:])


    def read_tape(self):
        """Returns the symbol or symbol block at the current head_pos"""
        # Single symbol head
        if self.head_shape is None:
            # Use 0 as the default value if the tape does not contain a symbol at the head
            if self.head_pos not in self.tape:
                self.tape[self.head_pos] = 0
            return self.tape[self.head_pos]
        # Multi symbol head
        else:
            symbol = np.empty(self.head_shape)
            points = cartesian_prod(*[list(range(i)) for i in self.head_shape])
            for point in points:
                point = tuple(point)
                tape_pos = tuple(np.array(self.head_pos) + point)
                if tape_pos not in self.tape:
                    symbol[point] = 0
                else:
                    symbol[point] = self.tape[tape_pos]
            return to_tuple(symbol)


    def write_tape(self, symbol):
        """Write the symbol or symbol block at the current head_pos"""
        # Single symbol head
        if type(symbol) not in (tuple, list, np.ndarray):
            self.tape[self.head_pos] = symbol
        # Multi symbol head
        else:
            symbol = np.array(symbol)
            points = cartesian_prod(*[list(range(i)) for i in symbol.shape])
            for point in points:
                point = tuple(point)
                tape_pos = tuple(np.array(self.head_pos) + point)
                self.tape[tape_pos] = symbol[point]


    def step(self):
        """Iterate the Turing Machine by one full step. Code is independent of reading and writing the tape."""

        # Current state and symbol
        # If the transition is a dict, this value is the key
        # If the transition is an ndarray, this value is used as an index
        state_and_symbol = (self.state, self.read_tape())

        # Determine next state, symbol, and how the head should move
        # Halt the machine if a transition is not defined
        if type(self.trans) != dict and np.less(state_and_symbol, self.trans.shape[:2]).all():
            new_state, new_symbol, *move = self.trans[state_and_symbol]
        elif type(self.trans) == dict and state_and_symbol in self.trans:
            new_state, new_symbol, move = self.trans[state_and_symbol]
        else:
            new_state = TM.HALT_STATE
            new_symbol = state_and_symbol[1]
            move = (0,) * self.N

        # Swap state, modify the tape, move the head
        self.state = new_state
        self.state_history.append(self.state)
        self.write_tape(new_symbol)
        self.head_pos = tuple(self.head_pos[i] + move[i] for i in range(len(move)))
        self.head_pos_history.append(self.head_pos)


    def get_tape_as_array(self):
        """Returns the current Turing tape as an array"""
        keys = np.array(list(self.tape.keys()))
        mins = np.min(keys, axis=0)
        maxs = np.max(keys, axis=0)
        tape = np.zeros(maxs-mins+1, int)
        for pos in self.tape.keys():
            symbol = self.tape[pos]
            i = tuple(pos - mins)
            tape[i] = symbol
        return tape


    def __str__(self):
        string = ''
        for r in self.get_tape_as_array():
            string += ''.join(map(str, r)) + '\n'
        return string


    def run(self, steps):
        """Runs the machine until the halt state or the given number of steps is reached. Returns the tape as an array."""
        for _ in range(steps):
            self.step()
            if self.state == TM.HALT_STATE:
                break
        return self.get_tape_as_array()






if __name__ == '__main__':

    # trans = [
    #     ['U', 0, 'R', 1, ( 1,  0)],
    #     ['R', 0, 'D', 1, ( 0, -1)],
    #     ['D', 0, 'L', 1, (-1,  0)],
    #     ['L', 0, 'U', 1, ( 0,  1)],
    #     ['U', 1, 'L', 0, (-1,  0)],
    #     ['R', 1, 'U', 0, ( 0,  1)],
    #     ['D', 1, 'R', 0, ( 1,  0)],
    #     ['L', 1, 'D', 0, ( 0, -1)],
    # ]
    # tm = TM(trans, state='U')
    # tape = tm(11000)
    # print(tm)

    trans = np.empty((4,2,4), int)
    trans[(0, 0)] = [1, 1,  1,  0]
    trans[(1, 0)] = [2, 1,  0, -1]
    trans[(2, 0)] = [3, 1, -1,  0]
    trans[(3, 0)] = [0, 1,  0,  1]
    trans[(0, 1)] = [3, 0, -1,  0]
    trans[(1, 1)] = [0, 0,  0,  1]
    trans[(2, 1)] = [1, 0,  1,  0]
    trans[(3, 1)] = [2, 0,  0, -1]
    tm = TM(trans)
    tape = tm.run(11000)
    print(tm)
    # print(trans)




    # X=TM.ANY
    # trans = [
    #     ['start', [[X,X,X],[X,0,X],[X,X,X]], 'start', [[X,X,X],[X,1,X],[X,X,X]], (+0, +1)],
    # ]
    # tm = TM(trans)
    # tape = tm(10)
    # print(tm)