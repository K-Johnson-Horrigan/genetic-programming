
from node import *

def builder(ops, terminals, reps=1, data=None):

    data = [[Node(terminals[0])]] if data is None else data

    data.append([])

    for base in data[-2]:
        l = len([n for n in base.nodes() if len(n) == 0])
        # node to replace
        for i in range(l):
            for op in ops:
                for op_terminals in itertools.product(terminals, repeat=2):
                    op_terminals = [Node(op_terminal) for op_terminal in op_terminals]
                    new_node = Node(op, op_terminals)
                    root = base.copy()
                    # Get a list of all terminals
                    nodes = [n for n in root.nodes() if len(n) == 0]
                    # Only replace nodes that match the first
                    if nodes[i].value == terminals[0]:
                        # continue
                        nodes[i].replace(new_node)
                        data[-1].append(root)

    if reps > 1:
        builder(ops=ops, terminals=terminals, reps=reps - 1, data=data)

    return data


# def finder(y, *x, reps=2, ops=('+', '-', '*', '/', '**')):
#     terminals = ['x_'+str(i) for i in range(len(x))]
#     y = np.array(y)
#     x = [np.array(x_i) for x_i in x]
#     built = builder(ops=ops, terminals=terminals, reps=reps)
#     for node in built[-1]:
#         y_n = node(*x)
#         if (y_n > 1e10).any():
#             print('Found:', y_n, node)


def finder(*x, reps=2, ops=('+', '-', '*', '/', '**')):
    terminals = ['x_' + str(i) for i in range(len(x))]
    x = [np.array(x_i) for x_i in x]
    built = builder(ops=ops, terminals=terminals, reps=reps)

    # table

    all_values = []
    all_counts = []

    d = {'{:04b}'.format(i) : [0]*len(built) for i in range(16)}

    # ys = []
    for rep in range(len(built)):
        # Calculate values of all nodes
        # ys.append([])
        ys = []
        for node in built[rep]:
            y = node(*x)
            ys.append(y)
            # if ((y != 1) & (y != 0)).any():
            #     print('Found:', y, node)

        # Convert nodes into LaTeX table
        vals, counts = np.unique(ys, axis=0, return_counts=True)
        # all_values.append(vals)
        # all_counts.append(counts)

    # for rep in
    # for i in range(len(all_values)):
        for v, c in zip(vals, counts):
            # if ((v == 1) | (v == 0)).all():
            bit_str = "".join(map(str, map(int, v)))
            if bit_str in d:
                d[bit_str][rep] = c

    for key in d.keys():
        s = key + '' + ' & '.join(map(str, d[key])) + ' \\\\'
        print(s)

    # print(d)

    # b_count = 0
    # for v,c in zip(vals, counts):
    #     if ((v == 1) | (v == 0)).all():
    #     # if True:
    #         b_count += c
    #         s = f'& {"".join(map(str,map(int,v)))} & {c} \\\\'
    #         # s = f'& {v} & {c} \\\\'
    #         print(s)
    # print(f'& other & {sum(counts) - b_count} \\\\')
    # print(f'& total & {sum(counts)} \\\\')

if __name__ == '__main__':

    # f = (x+x).to_tree()

    # b = finder([[x]], ['+','-','*','/','**'], ['x_0','x_1'], 2)
    # b = builder(['+', '-', '*', '/'], ['x_0', 'x_1'], 2)

    # for i in b[-1]:
    #     print(i)

    # Boolean
    # finder(
    #     [1,1,math.inf,1],
    #     [0,0,1,1],
    #     [0,1,0,1],
    #     reps=3
    # )

    finder(
        [0, 0, 1, 1],
        [0, 1, 0, 1],
        reps=3
    )
