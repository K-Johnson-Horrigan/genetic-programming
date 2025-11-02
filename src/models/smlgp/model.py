import numpy as np


class Linear:
    """
    Class representing a linear program during execution.

    """

    PC_INDEX = 0
    LINE_LENGTH = 4
    MAX_VALUE = 256

    DEFAULT_OPS = (
        'STOP',
        'LOAD',
        'STORE',
        'ADD',
        'SUB',
        'IFEQ',
        'RAND',
        'DEL',
    )

    # Addressing modes
    DEFAULT_ADDR_MODES = (
        'IMMEDIATE',
        'REGS_DIRECT',
        'REGS_INDIRECT',
        'CODE_DIRECT',
        'CODE_INDIRECT',
    )

    def __init__(self, mem, valid_ops=DEFAULT_OPS):
        self.mem = mem
        self.regs = self.mem[0]
        self.code = self.mem[1]

        # Create a list and dict to map between op code strings and values
        self.valid_ops_list = valid_ops
        self.valid_ops_dict = {v:u for u,v in enumerate(self.valid_ops_list)}

        # Create a list and dict to map between addr mode strings and values
        self.valid_addr_modes_list = list(Linear.DEFAULT_ADDR_MODES)
        self.valid_addr_modes_list += sum([[f'MEM{i}_DIRECT', f'MEM{i}_INDIRECT'] for i in range(2, len(self.mem))], [])
        self.valid_addr_modes_dict = {v:u for u,v in enumerate(self.valid_addr_modes_list)}

        # Mem can also be passed as containing strings
        valid_dict = self.valid_ops_dict | self.valid_addr_modes_dict
        for i, mem_bank in enumerate(mem):
            for j, value in enumerate(mem_bank):
                if type(value) == str:
                    self.mem[i][j] = valid_dict[value]


    def step(self):

        # Fetch the current line to be executed and unpack
        # Program counter loops back to start
        pc = int(self.regs[Linear.PC_INDEX])
        code_line = [self.code[(pc + i) % len(self.code)] for i in range(Linear.LINE_LENGTH)]
        op_code, target_reg, operand_spec, addr_mode = code_line

        # Values are modified to always be valid references
        op_code    = int(op_code)    % len(self.valid_ops_dict)
        target_reg = int(target_reg) % len(self.regs) if self.valid_ops_list[op_code] != 'RAND' else int(target_reg)
        addr_mode  = int(addr_mode)  % (len(self.mem) * 2 + 1)

        # Fetch the operand
        # Value of 0 is IMMEDIATE
        # Odd numbers are DIRECT
        # Even values are INDIRECT
        # if addr_mode % 2 == 1:
        #     mem_index = (addr_mode - 1) // 2
        #     operand = self.mem[mem_index][int(operand_spec) % len(self.mem[mem_index])]
        # elif addr_mode % 2 == 0 and addr_mode != 0:
        #     mem_index = (addr_mode - 2) // 2
        #     operand = self.mem[mem_index][self.regs[int(operand_spec) % len(self.regs)] % len(self.mem[mem_index])]
        # else:
        #     operand = operand_spec

        # mem_pointer

        # Fetch the operand
        if addr_mode % 2 == 1:
            i = (addr_mode - 1) // 2
            j = int(operand_spec) % len(self.mem[i])
        elif addr_mode % 2 == 0 and addr_mode != 0:
            i = (addr_mode - 2) // 2
            j = self.regs[int(operand_spec) % len(self.regs)] % len(self.mem[i])
        else:
            i = 1
            j = (pc + 2) % len(self.mem[i])
        operand = self.mem[i][j]

        # Increment program counter
        self.regs[Linear.PC_INDEX] += Linear.LINE_LENGTH

        # Perform the operation
        match self.valid_ops_list[op_code]:
            case 'STOP': return True
            case 'STORE': self.mem[i][j] = self.regs[target_reg]
            case 'LOAD': self.regs[target_reg] =                          operand  % Linear.MAX_VALUE
            case 'ADD':  self.regs[target_reg] = (self.regs[target_reg] + operand) % Linear.MAX_VALUE
            case 'SUB':  self.regs[target_reg] = (self.regs[target_reg] - operand) % Linear.MAX_VALUE
            case 'MUL':  self.regs[target_reg] = (self.regs[target_reg] * operand) % Linear.MAX_VALUE
            case 'DIV':  self.regs[target_reg] = (self.regs[target_reg] / operand) % Linear.MAX_VALUE
            case 'IFEQ':
                if self.regs[target_reg] != operand:
                    self.regs[Linear.PC_INDEX] += Linear.LINE_LENGTH
            case 'RAND':
                low  = min(0, target_reg)
                high = max(0, target_reg)
                self.mem[i][j] = np.random.randint(low, high + 1)
            case 'DEL': del self.mem[i][j]
            # case 'DUPE':
        return False


    def run(self, steps):
        """Run the machine for the given number of steps or until it reaches stop"""
        for _ in range(steps):
            if self.step():
                break
        return self


    def __str__(self):
        string = ''
        for i,mem in enumerate(self.mem):
            string += f'self.mem[{i}]'
            if i == 0: string += ' (REGISTERS)'
            elif i == 1: string += ' (PROGRAM)'
            string += '\n'
            step_size = 1 if i == 0 else Linear.LINE_LENGTH
            for pc in range(0, len(mem), step_size):
                line = [mem[(pc + j) % len(mem)] for j in range(step_size)]
                # Line index
                string += f'  {pc:2} │ '
                # Line values
                string += ', '.join([f'{j:2}' for j in line])
                # Formated line as code
                if step_size != 1:
                    op_code, target_reg, operand_spec, addr_mode = line
                    op_code = int(op_code) % len(self.valid_ops_dict)
                    target_reg = int(target_reg) % len(self.regs) if self.valid_ops_list[op_code] != 'RAND' else int(target_reg)
                    addr_mode = int(addr_mode) % (len(self.mem) * 2 + 1)
                    # Replace number with constant reference
                    op_code = "'" + self.valid_ops_list[op_code] + "',"
                    addr_mode = self.valid_addr_modes_list[addr_mode] if addr_mode < len(self.valid_addr_modes_list) else addr_mode
                    string += f" │ {op_code:8} {target_reg:2}, {operand_spec:2}, '{addr_mode}',"
                string += '\n'
        return string


if __name__ == '__main__':
    pass

    ## Mutate ##
    # code = [[
    #     0,
    #     0,
    # ],[
    #     Linear.RAND, 64, 1, Linear.VARS_DIRECT,
    #     Linear.RAND, 64, 1, Linear.MEM2_INDIRECT,
    # ],[
    #     0, 0, 0, 0,
    #     0, 0, 0, 0,
    # ]]

    ## Self-Rep / Crossover / Mutation ##
    # code = [[
    #     0, # PC
    #     0, # Random value
    #     0, # Copy pointer
    #     0, # Temp
    # ],[
    #     Linear.RAND,  1,  1, Linear.VARS_DIRECT,   # Generate random value
    #     Linear.IFEQ,  1,  0, Linear.IMMEDIATE,     # Execute next line if random value is 0
    #     Linear.LOAD,  3,  2, Linear.CODE_INDIRECT, # Load temp value from MEM2
    #     Linear.IFEQ,  1,  0, Linear.IMMEDIATE,     # Execute next line if random value is 0
    #     Linear.STORE, 3,  2, Linear.MEM2_INDIRECT, # Store temp value into MEM2
    #     Linear.ADD,   2,  1, Linear.IMMEDIATE,     # Increment copy pointer
    # ],[
    #     32, 32, 32, 32,
    #     32, 32, 32, 32,
    #     32, 32, 32, 32,
    #     32, 32, 32, 32,
    #     32, 32, 32, 32,
    #     32, 32, 32, 32,
    # ]]

    ## Multiply ##
    # code = [[
    #     0,
    #     3,
    #     5,
    #     0,
    # ],[
    #     'IFEQ', 2,  4, 'CODE_DIRECT',
    #     'STOP', 2,  9, 'REGS_DIRECT',
    #     'SUB',  2, 15, 'CODE_DIRECT',
    #     'ADD',  3, 13, 'REGS_DIRECT',
    # ]]

    ## Evolved Self Rep ##
    # code = [
    #     [Linear.SUB   ,  1 ,  3 , Linear.IMMEDIATE],
    #     [Linear.LOAD  ,  2 ,  4 , Linear.MEM_INDIRECT],
    #     [Linear.STORE ,  2 , 10 , Linear.OUT_INDIRECT],
    #     # [Linear.IFEQ  ,  1 ,  8 , Linear.MEM_INDIRECT],
    #     [Linear.SUB   ,  1 ,  3 , Linear.OUT_DIRECT],
    # ]

    ## One Point Crossover ##
    # code = [[
    #     0, # PC
    #     0, # Copy pointer
    #     0, # Temp
    # ],[
    #     Linear.IFEQ,  1,  0, Linear.IMMEDIATE,     # Check if copy pointer is 0
    #     Linear.RAND,4*8,  1, Linear.VARS_DIRECT,   # Randomly move the copy pointer
    #     Linear.LOAD,  2,  1, Linear.MEM2_INDIRECT, # Load temp value from MEM2
    #     Linear.STORE, 2,  1, Linear.MEM3_INDIRECT, # Store temp value into MEM3
    #     Linear.ADD,   1,  1, Linear.IMMEDIATE,     # Increment copy pointer
    #     Linear.IFEQ,  1,4*8, Linear.IMMEDIATE,     # Check if copy pointer is at last position
    #     Linear.STOP,  0,  0, Linear.IMMEDIATE,     # End execution
    # ],
    #     [1] * 32,
    #     [2] * 32,
    # ]

    l = Linear(code)
    print(l)
    l.run(100)
    print(l)
    # l.run(4000)
    # print(l)

