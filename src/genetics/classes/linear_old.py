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
        'RAND',
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
        'REG_DIRECT',
        'REG_INDIRECT',
        'OUT_DIRECT',
        'OUT_INDIRECT',
        'MEM_DIRECT',
        'MEM_INDIRECT',
    )
    IMMEDIATE    = VALID_ADDR_MODES.index('IMMEDIATE')    if 'IMMEDIATE'    in VALID_ADDR_MODES else None
    REG_DIRECT   = VALID_ADDR_MODES.index('REG_DIRECT')   if 'REG_DIRECT'   in VALID_ADDR_MODES else None
    REG_INDIRECT = VALID_ADDR_MODES.index('REG_INDIRECT') if 'REG_INDIRECT' in VALID_ADDR_MODES else None
    OUT_DIRECT   = VALID_ADDR_MODES.index('OUT_DIRECT')   if 'OUT_DIRECT'   in VALID_ADDR_MODES else None
    OUT_INDIRECT = VALID_ADDR_MODES.index('OUT_INDIRECT') if 'OUT_INDIRECT' in VALID_ADDR_MODES else None
    MEM_DIRECT   = VALID_ADDR_MODES.index('MEM_DIRECT')   if 'MEM_DIRECT'   in VALID_ADDR_MODES else None
    MEM_INDIRECT = VALID_ADDR_MODES.index('MEM_INDIRECT') if 'MEM_INDIRECT' in VALID_ADDR_MODES else None


    def __init__(self, code, input_regs=tuple(), num_output_regs=0):
        if type(code[0]) in (tuple, list, np.ndarray):
            self.mem = sum([list(line) for line in code], [])
        else:
            self.mem = list(code)

        self.reg = [0] + list(input_regs)
        self.out = [0] * num_output_regs
        # self.mem = [self.num_reg+1] + list(input_regs) + [0]*num_output_regs + sum(code, [])


    def step(self):

        # Fetch the current line to be executed and unpack
        # Program counter loops back to start
        pc = self.reg[Linear.PROGRAM_COUNTER_REGISTER_INDEX]
        op_code      = self.mem[(pc+0) % len(self.mem)]
        target_reg   = self.mem[(pc+1) % len(self.mem)]
        operand_spec = self.mem[(pc+2) % len(self.mem)]
        addr_mode    = self.mem[(pc+3) % len(self.mem)]

        # Values are modified to always be valid references
        op_code    = int(op_code)    % len(Linear.VALID_OPS)
        target_reg = int(target_reg) % len(self.reg)
        addr_mode  = int(addr_mode)  % len(Linear.VALID_ADDR_MODES)

        # Fetch the operand
        match addr_mode:
            case None: raise BaseException('invalid address mode')
            case Linear.IMMEDIATE:     operand =                       operand_spec
            case Linear.REG_DIRECT:    operand = self.reg[         int(operand_spec)                  % len(self.reg)]
            case Linear.REG_INDIRECT:  operand = self.reg[self.reg[int(operand_spec) % len(self.reg)] % len(self.reg)]
            case Linear.OUT_DIRECT:    operand = self.out[         int(operand_spec)                  % len(self.out)]
            case Linear.OUT_INDIRECT:  operand = self.out[self.reg[int(operand_spec) % len(self.reg)] % len(self.out)]
            case Linear.MEM_DIRECT:    operand = self.mem[         int(operand_spec)                  % len(self.mem)]
            case Linear.MEM_INDIRECT:  operand = self.mem[self.reg[int(operand_spec) % len(self.reg)] % len(self.mem)]
            case _:                    operand = None

        # Increment register
        self.reg[Linear.PROGRAM_COUNTER_REGISTER_INDEX] += Linear.LINE_LENGTH

        # Preform the operation
        match op_code:
            case None: raise BaseException('invalid operation')
            case Linear.STOP: return True
            case Linear.STORE:  # STORE must save to mem instead of fetching
                match addr_mode:
                    case Linear.REG_DIRECT:   self.reg[         int(operand_spec)                  % len(self.reg)] = self.reg[target_reg]
                    case Linear.REG_INDIRECT: self.reg[self.reg[int(operand_spec) % len(self.reg)] % len(self.reg)] = self.reg[target_reg]
                    case Linear.OUT_DIRECT:   self.out[         int(operand_spec)                  % len(self.out)] = self.reg[target_reg]
                    case Linear.OUT_INDIRECT: self.out[self.reg[int(operand_spec) % len(self.reg)] % len(self.out)] = self.reg[target_reg]
                    case Linear.MEM_DIRECT:   self.mem[         int(operand_spec)                  % len(self.mem)] = self.reg[target_reg]
                    case Linear.MEM_INDIRECT: self.mem[self.reg[int(operand_spec) % len(self.reg)] % len(self.mem)] = self.reg[target_reg]
            case Linear.LOAD: self.reg[target_reg] =                         operand  % Linear.MAX_VALUE
            case Linear.ADD:  self.reg[target_reg] = (self.reg[target_reg] + operand) % Linear.MAX_VALUE
            case Linear.SUB:  self.reg[target_reg] = (self.reg[target_reg] - operand) % Linear.MAX_VALUE
            case Linear.MUL:  self.reg[target_reg] = (self.reg[target_reg] * operand) % Linear.MAX_VALUE
            case Linear.DIV:  self.reg[target_reg] = (self.reg[target_reg] / operand) % Linear.MAX_VALUE
            case Linear.IFEQ:
                if self.reg[target_reg] != operand:
                    self.reg[Linear.PROGRAM_COUNTER_REGISTER_INDEX] += Linear.LINE_LENGTH
            case Linear.RAND:
                low  = min(0, target_reg)
                high = max(0, target_reg)
                random_val = np.random.randint(low, high + 1)
                # self.reg[target_reg] = np.random.randint(low, high + 1)
                match addr_mode:
                    case Linear.REG_DIRECT:   self.reg[         int(operand_spec)                  % len(self.reg)] = random_val
                    case Linear.REG_INDIRECT: self.reg[self.reg[int(operand_spec) % len(self.reg)] % len(self.reg)] = random_val
                    case Linear.OUT_DIRECT:   self.out[         int(operand_spec)                  % len(self.out)] = random_val
                    case Linear.OUT_INDIRECT: self.out[self.reg[int(operand_spec) % len(self.reg)] % len(self.out)] = random_val
                    case Linear.MEM_DIRECT:   self.mem[         int(operand_spec)                  % len(self.mem)] = random_val
                    case Linear.MEM_INDIRECT: self.mem[self.reg[int(operand_spec) % len(self.reg)] % len(self.mem)] = random_val
        return False


    def run(self, steps):
        """Run the machine for the given number of steps or until it reaches stop"""
        for _ in range(steps):
            if self.step():
                break


    def __str__(self):
        strings = ''
        strings += 'REGISTERS\n'
        for i,reg in enumerate(self.reg):
            strings += '  {:2} │ {:3}\n'.format(i, reg)
        for label,mode in [['PROGRAM MEMORY', self.mem], ['OUTPUT MEMORY', self.out]]:
            # strings.append(f'{label} [{len(mode)}]')
            strings += f'{label}\n'
            for pc in range(0, len(mode), Linear.LINE_LENGTH):
                strings += f'  {pc:2}'
                op_code, target_reg, operand_spec, addr_mode = mode[pc:pc + Linear.LINE_LENGTH]
                strings += ' │ [{: 3},{: 3},{: 3},{: 3}],'.format(op_code, target_reg, operand_spec, addr_mode)
                op_code    = int(op_code)    % len(Linear.VALID_OPS)
                target_reg = int(target_reg) % len(self.reg)
                addr_mode  = int(addr_mode)  % len(Linear.VALID_ADDR_MODES)
                op_code = Linear.VALID_OPS[op_code]
                addr_mode = Linear.VALID_ADDR_MODES[addr_mode]
                strings += ' │ [Linear.{:5}, {:2}, {:2}, Linear.{}],'.format(op_code, target_reg, operand_spec, addr_mode)
                strings += '\n'

        return strings


    # def simplified_str(self):
    #     """Convert the program into a string"""
    #     strings = [f'mem[0] = {self.mem[0]}  # Program Counter\n']
    #     for pc in range(1, self.num_reg+1):
    #         strings.append(f'mem[{pc}] = {self.mem[pc]}  # Register {pc}\n')
    #     for pc in range(self.num_reg+1, len(self.mem), Linear.LINE_LENGTH):
    #         op, target_reg, operand_spec, addr_mode = self.mem[pc:pc + Linear.LINE_LENGTH]
    #         target_reg = f'mem[{target_reg % len(self.mem)}]'
    #         match addr_mode:
    #             case Linear.IMMEDIATE: operand = operand_spec
    #             case Linear.DIRECT:    operand = f'mem[{operand_spec % len(self.mem)}]'
    #             case Linear.INDIRECT:  operand = f'mem[mem[{operand_spec}]]'
    #             case _:                operand = 'UNKNOWN ADDRESS MODE'
    #         match op:
    #             case Linear.STOP:  line = 'STOP\n'
    #             case Linear.STORE: line = f'{operand} = {target_reg}\n'
    #             case Linear.LOAD:  line = f'{target_reg} = {operand}\n'
    #             case Linear.ADD:   line = f'{target_reg} += {operand}\n'
    #             case Linear.SUB:   line = f'{target_reg} -= {operand}\n'
    #             case Linear.MUL:   line = f'{target_reg} *= {operand}\n'
    #             # case Linear.DIV:   line = f'{target_reg} /= {operand}\n'
    #             case Linear.IFEQ:  line = f'if {target_reg} == {operand}:\n\t'
    #             case _:            line = 'UNKNOWN OPERATION\n'
    #         strings.append(line)
    #     return ''.join(strings)




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
    #     [Linear.RAND, 64, 1, Linear.REG_DIRECT],
    #     [Linear.RAND, 64, 1, Linear.OUT_INDIRECT],
    # ]

    # Crossover
    code = [
        [Linear.RAND, 64, 1, Linear.REG_DIRECT],
        [Linear.RAND, 64, 1, Linear.OUT_INDIRECT],
    ]


    l = Linear(code, [0]*65, 4)
    l.run(2)
    print(l)

