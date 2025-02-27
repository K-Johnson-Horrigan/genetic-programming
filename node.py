import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import random
import math
from math import sin
import networkx as nx
from matplotlib import patheffects



from utils import plot_nodes


class Node:
    """A basic class for genetic programming. A Node holds a single value and points to zero or more children Nodes."""

    # All possible values for a node and the number of children it can have
    valid_ops = {
        '---': 1,
        '+': 2,
        '-': 2,
        '*': 2,
        '/': 2,
        '**': 2,
        'abs': 1,
        '==': 2,
        'if_then_else': 3,
        '&': 2,
        '|': 2,
        '<': 2,
        '>': 2,
        '<=': 2,
        '>=': 2,
        'min': 2,
        'max': 2,
        '%': 2,
        'sin': 1,
    }

    terminals = [
        'x',
    ]

    def __init__(self, value, children=None):
        self.parent = None
        self.parents = []
        # Cast int to a Node containing only valid terminals
        # if type(value) == int:
        #     value = Node.const(value)
        # If the value is already a node use its value so that Nodes can be cast to a Node
        # This also allows for copies of a Node to be made through casting
        if type(value) == Node:
            # self.children = value.copy().children
            self.children = value.children
            self.value = value.value
        else:
            self.value = value
            self.children = children if children is not None else []

    #
    # Children
    #

    @property
    def children(self):
        return self._children

    @children.setter
    def children(self, children):
        """Setting a child also sets the parent of the child"""
        for child in children:
            child.parent = self
        self._children = children

    def __len__(self): return len(self.children)
    def __getitem__(self, i): return self.children[i]
    def __setitem__(self, i, value): self.children[i] = value
    def __iter__(self): yield from self.children

    def nodes(self, node_list=None):
        """Returns a list of all nodes"""
        if node_list is None: node_list = []
        err = [s for s in node_list if s is self]
        if err:
            print('ERROR', self, node_list, err)
        node_list.append(self)
        for child in self:
            child.nodes(node_list)
        return node_list

    def depth(self):
        return max([0] + [1 + child.depth() for child in self.children])

    def node_depth(self):
        return 0 if self.parent is None else self.parent.node_depth() + 1

    def root(self):
        """Returns the root or parent of the tree"""
        return self if self.parent is None else self.parent.root()

    def replace(self, new_node):
        """Replaces this node and all children with a new branch"""
        # Create a copy of the new node
        new_node = new_node.copy()
        # Return the new node if self is the root of the tree
        if self.parent is None: return new_node
        # Parent's index for self
        parent_index = self.parent.children.index(self)
        # Replace the parent's reference to self
        self.parent[parent_index] = new_node
        # Replace the new Node's reference to parent
        new_node.parent = self.parent
        # Remove self reference to parent
        self.parent = None
        # Return the full new tree
        return new_node.root()

    #
    # Evaluation
    #

    def __call__(self, *x):
        """Calling evaluates the value of the entire tree"""

        # Simplify algebraically before evaluation
        # if algebraic:
        #     return self.simplify().evalf(subs={'x': x})

        # Evaluate as is
        # else:
        if type(self.value) is str:
            match self.value:
                case 'x': return x[0]
                case 'y': return x[1]
                case 'z': return x[2]
                case '+': return self[0](*x) + self[1](*x)
                case '-': return self[0](*x) - self[1](*x)
                case '*': return self[0](*x) * self[1](*x)
                case '**':
                    s0, s1 = self[0](*x), self[1](*x)
                    if s0 == 0 and s1 < 0:
                        return 1
                    else:
                        return self[0](*x) ** self[1](*x)
                case '/':
                    s0, s1 = self[0](*x), self[1](*x)
                    return 1 if s1 == 0 else s0 / s1
                case '|': return self[0](*x) or self[1](*x)
                case '&': return self[0](*x) and self[1](*x)
                case '<': return self[0](*x) < self[1](*x)
                case '>': return self[0](*x) > self[1](*x)
                case '<=': return self[0](*x) <= self[1](*x)
                case '>=': return self[0](*x) >= self[1](*x)
                case '==': return self[0](*x) == self[1](*x)
                case 'min': return min(self[0](*x), self[1](*x))
                case 'max': return max(self[0](*x), self[1](*x))
                case 'abs': return abs(self[0](*x))
                case 'if_then_else': return self[1](*x) if self[0](*x) else self[2](*x)
                case '%':  return self[0](*x) % self[1](*x)
                case '>>': return self[0](*x) >> self[1](*x)
                case '<<': return self[0](*x) << self[1](*x)
                case 'sin': return sin(self[0](*x))
                case _: return x[int(''.join([s for s in self.value if s.isdigit()]))]
        return self.value

    #
    # Utils
    #

    def __str__(self):
        if len(self) == 0:
            return str(self.value)
        elif self.value in ['+','-','*','/','**','&','|','%','>>','<<','<','>','<=','>=','==']:
            return f'({self[0]}{self.value}{self[1]})'
        else:
            return self.value + '(' + ','.join([str(child) for child in self]) + ')'

    def __repr__(self):
        return str(self)

    def simplify(self):
        return sp.sympify(self(sp.Symbol('x')))

    def copy(self):
        """Returns a recursive deepcopy of all Nodes"""
        return Node(self.value, [child.copy() for child in self])

    #
    # Native Python Conversion
    #

    @staticmethod
    def const(n):
        """A basic implementation to convert integers into limited trees"""
        x = Node('x')
        if n == 0:
            return x - x
        elif n == 1:
            return x / x
        elif n == -1:
            return Node.const(0) - Node.const(1)
        elif n < 0:
            return Node.const(-1) * Node.const(-n)
        else:
            return sum([Node.const(1) for _ in range(n-1)], Node.const(1))

    @staticmethod
    def op(operation, *operands):
        """Return a new Node from an operation on other Nodes"""
        # Convert operands to a list to be modified
        operands = list(operands)
        # Cast each operand to a Node or copy it if it is already a Node
        for i in range(len(operands)):
            if type(operands[i]) != Node:
                operands[i] = Node(operands[i])
            # else:
                # operands[i] = operands[i].copy()
        # Return a new Node with the operands as the children
        return Node(operation, operands)

    def      __add__(self, other): return Node.op('+',  self, other)
    def      __sub__(self, other): return Node.op('-',  self, other)
    def      __mul__(self, other): return Node.op('*',  self, other)
    def  __truediv__(self, other): return Node.op('/',  self, other)
    def      __pow__(self, other): return Node.op('**', self, other)
    def     __radd__(self, other): return Node.op('+',  other, self)
    def     __rsub__(self, other): return Node.op('-',  other, self)
    def     __rmul__(self, other): return Node.op('*',  other, self)
    def __rtruediv__(self, other): return Node.op('/',  other, self)
    def     __rpow__(self, other): return Node.op('**', other, self)

    def  __neg__(self): return 0 - self
    def  __and__(self, other): return self * other
    def __rand__(self, other): return other * self
    def   __or__(self, other): return self + other
    def  __ror__(self, other): return other + self
    def   __eq__(self, other): return 0 / (self - other)
    def  __abs__(self): return (self * self) ** (1 / 2)
    def   __lt__(self, other): return (1 - abs(self - other) / (self - other)) / 2
    def   __gt__(self, other): return (1 - abs(other - self) / (other - self)) / 2
    def   __le__(self, other): return (abs(other - self) / (other - self) + 1) / 2
    def   __ge__(self, other): return (abs(self - other) / (self - other) + 1) / 2
    def __lshift__(self, other): return self * 2 ** other
    def __rshift__(self, other): return self if other == 0 else ((self >> other-1) - (self >> other-1) % 2) / 2

    def __mod__(self, other):
        if other == 1:
            return Node(0)
        elif other == 2:
            return (1 - (-1) ** self) / 2
        else:
            k = int(math.log2(other))
            return (((self >> k-1) % 2) << k-1) + (self % 2**(k-1))



    @staticmethod
    def min(*args): return args[0] * (args[0] < args[1]) + args[1] * (args[0] >= args[1])
    @staticmethod
    def max(*args): return args[0] * (args[0] > args[1]) + args[1] * (args[0] <= args[1])
    @staticmethod
    def if_then_else(cond, if_true, if_false=None):
        if if_false is None:
            return cond * if_true
        else:
            return cond * if_true + (1 - cond) * if_false
    @staticmethod
    def read_bit(n, x): return ((x % n) - (x % (n/2))) / (n/2)

    #
    # Non-simplified
    #

    # def  __and__(self, other): return Node.op('&',  self, other)
    # def __rand__(self, other): return Node.op('&',  other, self)
    # def   __or__(self, other): return Node.op('|',  self, other)
    # def  __ror__(self, other): return Node.op('|',  other, self)
    # def   __eq__(self, other): return Node.op('==',  self, other)
    # def  __abs__(self): return Node.op('abs',  self)
    # def   __lt__(self, other): return Node.op('<',  self, other)
    # def   __gt__(self, other): return Node.op('>',  self, other)
    # def   __le__(self, other): return Node.op('<=',  self, other)
    # def   __ge__(self, other): return Node.op('>=',  self, other)
    # def __mod__ (self, other): return Node.op('%',self, other)
    # @staticmethod
    # def if_then_else(cond, if_true, if_false=None):
    #     return Node.op('if_then_else', cond, if_true, if_false)


    def disp(self, x0=0, x1=1, y=0, initial=True, verts=None, edges=None, pos=None):

        edges = [] if edges is None else edges
        verts = [self] if verts is None else verts
        pos = [(0,0)] if pos is None else pos
        index = [index for index,n in enumerate(verts) if n is self][0]

        x = x0 + 0.5 * (x1 - x0)

        xx = math.cos(math.pi * 2 * x) * y
        yy = math.sin(math.pi * 2 * x) * y


        if len(self) > 0:
            sub_y = y + 1
            for i, child in enumerate(self):
                # Check if the node already exists in the plot
                sub = [index for index,n in enumerate(verts) if n is child]
                if len(sub) > 0:
                    # plt.plot((xx, sub[0][1]), (yy, sub[0][2]), c='b')
                    # sub_xx = math.cos(math.pi * 2 * pos[child_index][0]) * pos[child_index][1]
                    # sub_yy = math.sin(math.pi * 2 * pos[child_index][0]) * pos[child_index][1]
                    # # Number of times the node is a child of this parent
                    # pars = len([p for p in sub[0][0].parents if p is self])
                    # if pars == 0:
                    #     plt.plot((xx, sub_xx), (yy, sub_yy), c='b', lw=200)
                    #     print(sub_xx, sub_yy)
                    # else:
                    #     plt.plot((xx, sub_xx), (yy, sub_yy), c='b', lw=200)
                    child_index = sub[0]
                    edges.append((index, child_index))
                # Plot children
                else:
                    child_index = len(verts)
                    verts.append(child)
                    pos.append(None)
                    # Position
                    sub_x0 = x0 +  i    / (len(self)) * (x1 - x0)
                    sub_x1 = x0 + (i+1) / (len(self)) * (x1 - x0)
                    sub_x, sub_y = child.disp(sub_x0, sub_x1, sub_y, False, verts, edges, pos)
                    sub_xx = math.cos(math.pi * 2 * sub_x) * sub_y
                    sub_yy = math.sin(math.pi * 2 * sub_x) * sub_y
                    # points
                    # x_points = xx, (sub_xx + xx) / 2, sub_xx
                    # y_points = yy, (sub_yy + yy) / 2, sub_yy
                    # plt.plot(x_points, y_points, marker='>', c='k')

                    pos[child_index] = (sub_xx, sub_yy)
                    edges.append((index, child_index))

        # plt.scatter(xx, yy, c='k', s=200)
        # plt.text(xx, yy, str(self.value), horizontalalignment='center', verticalalignment='center', color='white')

        if not initial:
            return x, y
        else:
            # print(len(saved))
            # plt.show()

            fig, ax = plt.subplots()
            # Create networkxs graph
            G = nx.MultiDiGraph()
            G.add_nodes_from(range(len(verts)))
            G.add_edges_from(edges)
            nx.set_node_attributes(G, {key: str(node.value) for key,node in enumerate(verts)}, 'label')
            G.nodes(data=True)

            # pos = nx.kamada_kawai_layout(G)
            # pos = nx.spring_layout(G)
            # pos = nx.spectral_layout(G)
            # pos = nx.arf_layout(G)
            # pos = nx.planar_layout(G)

            # Draw targets
            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=range(len(verts)),
                node_color='tab:blue',
            )

            # Draw vertex labels
            nx.draw_networkx_labels(
                G,
                pos,
                labels = {key: str(node.value) for key,node in enumerate(verts)},
                font_color="whitesmoke"
            )

            # Draw edges
            nx.draw_networkx_edges(
                G,
                pos,
                arrowstyle="->",
                arrowsize=10,
                # edge_color = range(G.number_of_edges()),
                # edge_cmap = plt.cm.gist_rainbow,
                width=2,
                alpha=0.5,
            )

            # ax.set_title(alg + ' Route ({:.3f} units)'.format(minimum))
            # ax.set_xlim([-.05, 1.05])
            # ax.set_ylim([-.05, 1.05])
            # ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

            plt.show()



def plot_node(node, x_linspace, **kwargs):
    """Plot all given nodes"""
    xs = np.linspace(*x_linspace)
    ys = [node(x) for x in xs]
    # ys0 = [int(x) >> 3 for x in xs]
    # ys0 = xs % 4
    # plt.scatter(xs, ys0, s=40)
    # plt.plot(xs, ys0)
    plt.xlim((0, 31))
    plt.ylim((0, 31))
    plt.scatter(xs, ys, s=10)
    plt.plot(xs, ys, ':')
    plt.plot([0,31],[0,31])
    plt.show()

def plot_nodes(nodes, x_linspace, **kwargs):
    """Plot all given nodes"""
    xs = np.linspace(*x_linspace)
    for node in nodes:
        ys = [node(x) for x in xs]
        # label = f'${str(kwargs['function'](sp.Symbol("x"))).replace("**","^")}$'
        plt.scatter(xs, ys)
        plt.plot(xs, ys)
    plt.show()




x = Node('x')


if __name__ == '__main__':
    y = Node('y')


    x = Node('x')
    f0 = x + 1
    f1 = f0 - 1
    f2 = f1 * f1
    f3 = f2 / f1
    f4 = f3 ** f2
    f = f4.copy()
    # f = f4


    # f = x + x

    # f = x % 8
    # f = f.copy()

    # f.nodes()

    # print(f[0][1].node_depth())

    # ReLu
    # f = Node.if_then_else(
    #     x >= 0,
    #     x
    # )
    # x*(0.5 + 0.5*(x**2)**0.5/x)

    # Collatz Conjecture
    # f = Node.if_then_else(
    #     x % 2,
    #     3 * x + 1,
    #     x / 2,
    # )
    # f = 2/4 + 7/4 * x +  (-2/4 + -5/4 * x) * (-1)**x
    # f = 2/4 + 7/4*x + (-2/4 + -5/4*x) * cos(pi * x)
    # f(n) = 2 / 4 + 7 / 4 * f(n-1) + (-2 / 4 + -5 / 4 * f(n-1)) * cos(pi * f(n-1))
    # i = 43
    # y = [i]
    # while i != 1:
    #     print(i)
    #     i = f(i)
    #     y.append(i)
    # y = np.array(y)
    # x = np.arange(len(y))
    # # Loop plot
    # xx = y.copy()
    # yy = y.copy()
    # yy[0] = 0
    # xx[1::2] = xx[0::2]
    # yy[2::2] = yy[1:-1:2]
    # plt.plot(xx, yy)
    # plt.scatter(xx,yy)
    # plt.axline((0, 1), (1, 4), ls=':')
    # plt.axline((0, 0), (1, 1/2), ls=':')
    # plt.axline((1, 0), (4, 1), ls=':')
    # plt.axline((0, 0), (1/2, 1), ls=':')
    # plt.plot()
    # plt.show()



    # plt.gca().set_yscale('log')
    f.disp()


    # print(f)
    # print(f.simplify())
    # plot_node(f, (0,31,32))
    # for _ in range(3):
    #     f = f(f)
    #     print(f.simplify())