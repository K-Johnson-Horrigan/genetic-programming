import numpy as np


class Linear:
    """
    Class representing a linear program during execution.

    """

    PROGRAM_COUNTER_REGISTER_INDEX = 0
    LINE_LENGTH = 4

    VALID_OPS = (
        'STOP',
        'LOAD',
        'STORE',
        'ADD',
        'SUB',
        'MUL',
        # 'DIV',
        'IFEQ',
    )
    STOP  = VALID_OPS.index('STOP')  if 'STOP'  in VALID_OPS else None
    LOAD  = VALID_OPS.index('LOAD')  if 'LOAD'  in VALID_OPS else None
    STORE = VALID_OPS.index('STORE') if 'STORE' in VALID_OPS else None
    ADD   = VALID_OPS.index('ADD')   if 'ADD'   in VALID_OPS else None
    SUB   = VALID_OPS.index('SUB')   if 'SUB'   in VALID_OPS else None
    MUL   = VALID_OPS.index('MUL')   if 'MUL'   in VALID_OPS else None
    DIV   = VALID_OPS.index('DIV')   if 'DIV'   in VALID_OPS else None
    IFEQ  = VALID_OPS.index('IFEQ')  if 'IFEQ'  in VALID_OPS else None

    # Addressing modes
    VALID_ADDR_MODES = (
        'IMMEDIATE',
        'DIRECT',
        'INDIRECT',
        'OUTPUT_DIRECT',
        'OUTPUT_INDIRECT',
        'CODE_DIRECT',
        'CODE_INDIRECT',
    )
    IMMEDIATE       = VALID_ADDR_MODES.index('IMMEDIATE') if 'IMMEDIATE' in VALID_ADDR_MODES else None
    DIRECT          = VALID_ADDR_MODES.index('DIRECT')    if 'DIRECT'    in VALID_ADDR_MODES else None
    INDIRECT        = VALID_ADDR_MODES.index('INDIRECT')  if 'INDIRECT'  in VALID_ADDR_MODES else None
    # INPUT_DIRECT    = VALID_ADDR_MODES.index('INPUT_DIRECT')    if 'INPUT_DIRECT'    in VALID_ADDR_MODES else None
    # INPUT_INDIRECT  = VALID_ADDR_MODES.index('INPUT_INDIRECT')  if 'INPUT_INDIRECT'  in VALID_ADDR_MODES else None
    OUTPUT_DIRECT   = VALID_ADDR_MODES.index('OUTPUT_DIRECT')   if 'OUTPUT_DIRECT'   in VALID_ADDR_MODES else None
    OUTPUT_INDIRECT = VALID_ADDR_MODES.index('OUTPUT_INDIRECT') if 'OUTPUT_INDIRECT' in VALID_ADDR_MODES else None
    CODE_DIRECT     = VALID_ADDR_MODES.index('CODE_DIRECT')     if 'CODE_DIRECT'     in VALID_ADDR_MODES else None
    CODE_INDIRECT   = VALID_ADDR_MODES.index('CODE_INDIRECT')   if 'CODE_INDIRECT'   in VALID_ADDR_MODES else None


    def __init__(self, code, input_regs=tuple(), num_output_regs=0):
        self.num_reg = len(input_regs) + num_output_regs
        self.num_reg = len(input_regs) + num_output_regs
        self.mem = [self.num_reg+1] + list(input_regs) + [0]*num_output_regs + sum(code, [])


    def step(self):

        # Fetch the current line to be executed and unpack
        # Program counter loops back to start
        pc = self.mem[Linear.PROGRAM_COUNTER_REGISTER_INDEX]
        pc = int(pc) % (len(self.mem) - 3)
        op, target_reg, operand_spec, addr_mode = self.mem[pc:pc+Linear.PROGRAM_COUNTER_REGISTER_INDEX]

        # Values are modified to always be valid references
        op = int(op) % len(Linear.VALID_OPS) if op != np.inf else 0
        target_reg = int(target_reg) % len(self.mem) if target_reg != np.inf else 0
        addr_mode = int(addr_mode) % len(Linear.VALID_ADDR_MODES) if addr_mode != np.inf else 0

        # Fetch the operand
        match addr_mode:
            case Linear.IMMEDIATE: operand = operand_spec
            case Linear.DIRECT:    operand = self.mem[int(operand_spec) % len(self.mem)]
            case Linear.INDIRECT:  operand = self.mem[self.mem[operand_spec]]
            # case Linear.RELATIVE:  operand = pc + operand_spec
            case _:                operand = None

        # Increment register
        self.mem[Linear.PROGRAM_COUNTER_REGISTER_INDEX] += Linear.LINE_LENGTH

        # Preform the operation
        match op:
            case Linear.STOP: return True
            case Linear.LOAD: self.mem[target_reg] = operand
            case Linear.STORE:
                # STORE must save to mem instead of fetching
                match addr_mode:
                    case Linear.DIRECT: self.mem[int(operand_spec) % len(self.mem)] = self.mem[target_reg]
                    case Linear.INDIRECT: self.mem[self.mem[operand_spec]] = self.mem[target_reg]
            case Linear.ADD: self.mem[target_reg] += operand
            case Linear.SUB: self.mem[target_reg] -= operand
            case Linear.MUL: self.mem[target_reg] *= operand
            case Linear.DIV: self.mem[target_reg] /= operand
            case Linear.IFEQ:
                if self.mem[target_reg] != operand:
                    self.mem[Linear.PROGRAM_COUNTER_REGISTER_INDEX] += Linear.LINE_LENGTH

        return False


    def run(self, steps):
        """Run the machine for the given number of steps or until it reaches stop"""
        for _ in range(steps):
            if self.step():
                break


    def __str__(self):
        strings = []
        for pc in range(self.num_reg+1, len(self.mem), Linear.LINE_LENGTH):
            op, target_reg, operand_spec, addr_mode = self.mem[pc:pc + Linear.LINE_LENGTH]
            if op < len(Linear.VALID_OPS) and addr_mode < len(Linear.VALID_ADDR_MODES):
                op = Linear.VALID_OPS[op]
                addr_mode = Linear.VALID_ADDR_MODES[addr_mode]
            strings.append('{:5} {} {} {}'.format(op, target_reg, operand_spec, addr_mode))
        return '\n'.join(strings)


    def simplified_str(self):
        """Convert the program into a string"""
        strings = [f'mem[0] = {self.mem[0]}  # Program Counter\n']
        for pc in range(1, self.num_reg+1):
            strings.append(f'mem[{pc}] = {self.mem[pc]}  # Register {pc}\n')
        for pc in range(self.num_reg+1, len(self.mem), Linear.LINE_LENGTH):
            op, target_reg, operand_spec, addr_mode = self.mem[pc:pc + Linear.LINE_LENGTH]
            target_reg = f'mem[{target_reg % len(self.mem)}]'
            match addr_mode:
                case Linear.IMMEDIATE: operand = operand_spec
                case Linear.DIRECT:    operand = f'mem[{operand_spec % len(self.mem)}]'
                case Linear.INDIRECT:  operand = f'mem[mem[{operand_spec}]]'
                case _:                operand = 'UNKNOWN ADDRESS MODE'
            match op:
                case Linear.STOP:  line = 'STOP\n'
                case Linear.STORE: line = f'{operand} = {target_reg}\n'
                case Linear.LOAD:  line = f'{target_reg} = {operand}\n'
                case Linear.ADD:   line = f'{target_reg} += {operand}\n'
                case Linear.SUB:   line = f'{target_reg} -= {operand}\n'
                case Linear.MUL:   line = f'{target_reg} *= {operand}\n'
                # case Linear.DIV:   line = f'{target_reg} /= {operand}\n'
                case Linear.IFEQ:  line = f'if {target_reg} == {operand}:\n\t'
                case _:            line = 'UNKNOWN OPERATION\n'
            strings.append(line)
        return ''.join(strings)




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




    pc,a,b,t = 0,1,2,3

    code = [
        [Linear.LOAD,  a, pc, Linear.DIRECT], # Copy PC to value of a-pointer
        [Linear.LOAD,  t,  a, Linear.INDIRECT], # Copy a-pointer to temp
        [Linear.STORE, t,  b, Linear.INDIRECT], # Copy temp to b-pointer
        [Linear.ADD,   a,  1, Linear.IMMEDIATE], # move a-pointer to next value
        [Linear.ADD,   b,  1, Linear.IMMEDIATE], # Move b-pointer to next value
        [Linear.IFEQ,  b, 32, Linear.IMMEDIATE], # b is at the end
        [Linear.STOP, 0, 0, Linear.IMMEDIATE], # Stop
        [Linear.SUB,  pc, 4*7, Linear.IMMEDIATE], # Return to start
    ]
    l = Linear(code, [0,4,0], 4*len(code))

    a = 0

    # match a:
    #     case 0: print(0)
    #     case 0: print(2)
    #     case 1: print(1)

    print(l.simplified_str())

    l.run(200)

    print(l.simplified_str())

    print(l.mem)