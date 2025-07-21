import numpy as np


class Linear:
    """
    Class representing a linear program during execution.

    """

    PROGRAM_COUNTER_REGISTER_INDEX = 0
    LINE_LENGTH = 4
    MAX_VALUE = 256

    VALID_OPS = (
        'STOP',
        'LOAD',
        'STORE',
        'ADD',
        'SUB',
        # 'MUL',
        # 'DIV',
        'IFEQ',
        # 'RAND',
    )
    STOP  = VALID_OPS.index('STOP')  if 'STOP'  in VALID_OPS else None
    LOAD  = VALID_OPS.index('LOAD')  if 'LOAD'  in VALID_OPS else None
    STORE = VALID_OPS.index('STORE') if 'STORE' in VALID_OPS else None
    ADD   = VALID_OPS.index('ADD')   if 'ADD'   in VALID_OPS else None
    SUB   = VALID_OPS.index('SUB')   if 'SUB'   in VALID_OPS else None
    MUL   = VALID_OPS.index('MUL')   if 'MUL'   in VALID_OPS else None
    DIV   = VALID_OPS.index('DIV')   if 'DIV'   in VALID_OPS else None
    IFEQ  = VALID_OPS.index('IFEQ')  if 'IFEQ'  in VALID_OPS else None
    RAND  = VALID_OPS.index('RAND')  if 'RAND'  in VALID_OPS else None

    # Addressing modes
    VALID_ADDR_MODES = (
        'IMMEDIATE',
        'VARS_DIRECT',
        'VARS_INDIRECT',
        'CODE_DIRECT',
        'CODE_INDIRECT',
        'MEM2_DIRECT',
        'MEM2_INDIRECT',
    )
    IMMEDIATE     = VALID_ADDR_MODES.index('IMMEDIATE')     if 'IMMEDIATE'     in VALID_ADDR_MODES else None
    VARS_DIRECT   = VALID_ADDR_MODES.index('VARS_DIRECT')   if 'VARS_DIRECT'   in VALID_ADDR_MODES else None
    VARS_INDIRECT = VALID_ADDR_MODES.index('VARS_INDIRECT') if 'VARS_INDIRECT' in VALID_ADDR_MODES else None
    CODE_DIRECT   = VALID_ADDR_MODES.index('CODE_DIRECT')   if 'CODE_DIRECT'   in VALID_ADDR_MODES else None
    CODE_INDIRECT = VALID_ADDR_MODES.index('CODE_INDIRECT') if 'CODE_INDIRECT' in VALID_ADDR_MODES else None
    MEM2_DIRECT   = VALID_ADDR_MODES.index('MEM2_DIRECT')   if 'MEM2_DIRECT'   in VALID_ADDR_MODES else None
    MEM2_INDIRECT = VALID_ADDR_MODES.index('MEM2_INDIRECT') if 'MEM2_INDIRECT' in VALID_ADDR_MODES else None

    def __init__(self, mem):
        self.mem = mem
        self.vars = self.mem[0]
        self.code = self.mem[1]


    def step(self):

        # Fetch the current line to be executed and unpack
        # Program counter loops back to start
        pc = int(self.vars[Linear.PROGRAM_COUNTER_REGISTER_INDEX])
        code_line = [self.code[(pc + i) % len(self.code)] for i in range(Linear.LINE_LENGTH)]
        op_code, target_reg, operand_spec, addr_mode = code_line

        # Values are modified to always be valid references
        op_code    = int(op_code)    % len(Linear.VALID_OPS)
        target_reg = int(target_reg) % len(self.vars) if op_code != Linear.RAND else int(target_reg)
        addr_mode  = int(addr_mode)  % (len(self.mem) * 2 + 1)

        # Fetch the operand
        if addr_mode % 2 == 1:
            mem_index = (addr_mode - 1) // 2
            operand = self.mem[mem_index][int(operand_spec) % len(self.mem[mem_index])]
        elif addr_mode % 2 == 0 and addr_mode != 0:
            mem_index = (addr_mode - 2) // 2
            operand = self.mem[mem_index][self.vars[int(operand_spec) % len(self.vars)] % len(self.mem[mem_index])]
        else:
            operand = operand_spec

        # Increment register
        self.vars[Linear.PROGRAM_COUNTER_REGISTER_INDEX] += Linear.LINE_LENGTH

        # Preform the operation
        match op_code:
            case None: raise BaseException('invalid operation')
            case Linear.STOP: return True
            case Linear.STORE:  # STORE must save to mem instead of fetching
                if addr_mode % 2 == 1:
                    mem_index = (addr_mode - 1) // 2
                    self.mem[mem_index][int(operand_spec) % len(self.mem[mem_index])] = self.vars[target_reg]
                elif addr_mode % 2 == 0 and addr_mode != 0:
                    mem_index = (addr_mode - 2) // 2
                    self.mem[mem_index][self.vars[int(operand_spec) % len(self.vars)] % len(self.mem[mem_index])] = self.vars[target_reg]
            case Linear.LOAD: self.vars[target_reg] =                          operand  % Linear.MAX_VALUE
            case Linear.ADD:  self.vars[target_reg] = (self.vars[target_reg] + operand) % Linear.MAX_VALUE
            case Linear.SUB:  self.vars[target_reg] = (self.vars[target_reg] - operand) % Linear.MAX_VALUE
            case Linear.MUL:  self.vars[target_reg] = (self.vars[target_reg] * operand) % Linear.MAX_VALUE
            case Linear.DIV:  self.vars[target_reg] = (self.vars[target_reg] / operand) % Linear.MAX_VALUE
            case Linear.IFEQ:
                if self.vars[target_reg] != operand:
                    self.vars[Linear.PROGRAM_COUNTER_REGISTER_INDEX] += Linear.LINE_LENGTH
            case Linear.RAND:
                low  = min(0, target_reg)
                high = max(0, target_reg)
                random_val = np.random.randint(low, high + 1)
                if addr_mode % 2 == 1:
                    mem_index = (addr_mode - 1) // 2
                    self.mem[mem_index][int(operand_spec) % len(self.mem[mem_index])] = random_val
                elif addr_mode % 2 == 0 and addr_mode != 0:
                    mem_index = (addr_mode - 2) // 2
                    self.mem[mem_index][self.vars[int(operand_spec) % len(self.vars)] % len(self.mem[mem_index])] = random_val
        return False


    def run(self, steps):
        """Run the machine for the given number of steps or until it reaches stop"""
        for _ in range(steps):
            if self.step():
                break


    def __str__(self):
        strings = ''
        for i,mem in enumerate(self.mem):
            strings += f'self.mem[{i}]'
            if i == 0: strings += ' (VARIABLES)'
            elif i == 1: strings += ' (PROGRAM)'
            strings += '\n'
            step_size = 1 if i == 0 else Linear.LINE_LENGTH
            for pc in range(0, len(mem), step_size):
                line = [mem[(pc + j) % len(mem)] for j in range(step_size)]
                strings += f'  {pc:2} │ '
                # if step_size != 1: strings += '['
                strings += ', '.join([f'{j:2}' for j in line])
                # if step_size != 1: strings += '],'
                if i != 0:
                    op_code, target_reg, operand_spec, addr_mode = line
                    op_code    = int(op_code)    % len(Linear.VALID_OPS)
                    target_reg = int(target_reg) % len(self.vars) if op_code != Linear.RAND else int(target_reg)
                    addr_mode  = int(addr_mode)  % len(self.mem) * 2 + 1
                    # Replace number with constant reference
                    op_code = Linear.VALID_OPS[op_code]
                    addr_mode = Linear.VALID_ADDR_MODES[addr_mode] if addr_mode < len(Linear.VALID_ADDR_MODES) else addr_mode
                    strings += f' │ Linear.{op_code:5}, {target_reg:2}, {operand_spec:2}, Linear.{addr_mode},'
                strings += '\n'
        return strings


if __name__ == '__main__':

    # pc,a,b,t = 0,1,2,3
    # code = [
    #     [Linear.LOAD,  t, a, Linear.INDIRECT],
    #     [Linear.STORE, t, b, Linear.INDIRECT],
    #     [Linear.ADD,   a, 1, Linear.IMMEDIATE],
    #     [Linear.ADD,   b, 1, Linear.IMMEDIATE],
    #     [Linear.SUB,  pc, 4*5, Linear.IMMEDIATE],
    #     [Linear.STOP, 0, 0, Linear.IMMEDIATE] * 20
    # ]
    # l = Linear(code, [4,4*7], 1)


    # code = [
    #     [Linear.LOAD,  a, pc, Linear.DIRECT], # Copy PC to value of a-pointer
    #     [Linear.LOAD,  t,  a, Linear.INDIRECT], # Copy a-pointer to temp
    #     [Linear.STORE, t,  b, Linear.INDIRECT], # Copy temp to b-pointer
    #     [Linear.ADD,   a,  1, Linear.IMMEDIATE], # move a-pointer to next value
    #     [Linear.ADD,   b,  1, Linear.IMMEDIATE], # Move b-pointer to next value
    #     [Linear.IFEQ,  b, 32, Linear.IMMEDIATE], # b is at the end
    #     [Linear.STOP, 0, 0, Linear.IMMEDIATE], # Stop
    #     [Linear.SUB,  pc, 4*7, Linear.IMMEDIATE], # Return to start
    # ]

    # code = [
    #     [Linear.RAND, 64, 1, Linear.VARS_DIRECT],
    #     [Linear.RAND, 64, 1, Linear.OUT_INDIRECT],
    # ]

    # # Mutate
    # code = [[
    #     0, 0,
    # ],[
    #     Linear.RAND, 64, 1, Linear.VARS_DIRECT,
    #     Linear.RAND, 64, 1, Linear.MEM2_INDIRECT,
    # ],[
    #     0, 0, 0, 0,
    #     0, 0, 0, 0,
    # ]]

    # Self-Rep / Crossover / Mutation
    code = [[
        0, # PC
        0, # Random value
        0, # Copy pointer
        0, # Temp
    ],[
        Linear.RAND,  1,  1, Linear.VARS_DIRECT,   # Generate random value
        Linear.IFEQ,  1,  0, Linear.IMMEDIATE,     # Execute next line if random value is 0
        Linear.LOAD,  3,  2, Linear.CODE_INDIRECT, # Load temp value from MEM2
        Linear.IFEQ,  1,  0, Linear.IMMEDIATE,     # Execute next line if random value is 0
        Linear.STORE, 3,  2, Linear.MEM2_INDIRECT, # Store temp value into MEM2
        Linear.ADD,   2,  1, Linear.IMMEDIATE,     # Increment copy pointer
    ],[
        32, 32, 32, 32,
        32, 32, 32, 32,
        32, 32, 32, 32,
        32, 32, 32, 32,
        32, 32, 32, 32,
        32, 32, 32, 32,
    ]]

    l = Linear(code)
    l.run(6 * 4 * 6)
    print(l)

