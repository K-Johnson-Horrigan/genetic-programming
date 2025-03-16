import math
from math import sin, cos

import sympy as sp

from plot import plot_nodes, plot_tree


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
        'cos': 1,
    }

    terminals = [
        'x',
    ]

    def __init__(self, value, children=None):
        self.parent = None
        self.parents = []
        self.temp_index = None # Used when creating a list of all nodes to prevent repeats.
        # If the value is already a node use its value so that Nodes can be cast to a Node
        # This also allows for copies of a Node to be made through casting
        if type(value) == Node:
            self.children = value.copy().children
            # self.children = value.children
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
            child.parents.append(self) #FIXME remove unused parents
        self._children = children

    def __len__(self): return len(self.children)
    def __getitem__(self, i): return self.children[i]
    def __setitem__(self, i, value): self.children[i] = value
    def __iter__(self): yield from self.children

    def reset_index(self):
        if self.temp_index is not None:
            self.temp_index = None
            for child in self.children:
                child.reset_index()

    def nodes(self, node_list=None):
        """Returns a list of all nodes"""
        if node_list is None:
            node_list = []
            self.reset_index()
        if self.temp_index is None:
            self.temp_index = len(node_list)
            node_list.append(self)
            for child in self:
                child.nodes(node_list)
        return node_list

    def height(self):
        """Longest distance to a leaf"""
        return max([0] + [1 + child.height() for child in self.children])

    def depth(self):
        """Longest distance to the root"""
        return max([1] + [1 + parent.depth() for parent in self.parents])

    def root(self):
        """Returns the root Node of the tree"""
        return self if self.parent is None else self.parent.root()

    def copy(self):
        return Node.from_lists(*self.to_lists())

    def expanded_copy(self):
        """Returns a recursive deepcopy of all Nodes"""
        return Node(self.value, [child.copy() for child in self])

    def to_lists(self, verts=None, edges=None):
        """Returns lists representing the vertices and edges"""
        if verts is None:
            self.reset_index()
            verts, edges = [], []
        if self.temp_index is None:
            self.temp_index = len(verts)
            verts.append(self.value)
            for child in self.children:
                child.to_lists(verts, edges)
                edges.append((self.temp_index, child.temp_index))
        return verts, edges

    @staticmethod
    def from_lists(verts, edges):
        """Returns a Node tree from lists representing the vertices and edges"""
        nodes = [Node(vert) for vert in verts]
        for edge in edges:
            nodes[edge[0]]._children.append(nodes[edge[1]])
            nodes[edge[1]].parents.append(nodes[edge[0]])
        return nodes[0]

    def replace(self, new_node):
        """Replaces this node and all children with a new branch"""
        # Create a copy of the new node
        new_node = new_node.copy()
        # Return the new node if self is the root of the tree
        if self.parent is None: return new_node
        # Parent's index for self
        parent_index = self.parent.children.temp_index(self)
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
                case 'cos': return cos(self[0](*x))
                case _: return x[int(''.join([s for s in self.value if s.isdigit()]))]
        return self.value

    def simplify(self):
        return sp.sympify(self(sp.Symbol('x')))

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

    #
    # Native Python Conversion
    #

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
    def     __radd__(self, other): return Node.op('+',  other, self)
    def      __sub__(self, other): return Node.op('-',  self, other)
    def     __rsub__(self, other): return Node.op('-',  other, self)
    def      __mul__(self, other): return Node.op('*',  self, other)
    def     __rmul__(self, other): return Node.op('*',  other, self)
    def  __truediv__(self, other): return Node.op('/',  self, other)
    def __rtruediv__(self, other): return Node.op('/',  other, self)
    def      __pow__(self, other): return Node.op('**', self, other)
    def     __rpow__(self, other): return Node.op('**', other, self)

    def  __neg__(self): return Node.op('neg',  self)
    def  __and__(self, other): return Node.op('&',  self, other)
    def __rand__(self, other): return Node.op('&',  other, self)
    def   __or__(self, other): return Node.op('|',  self, other)
    def  __ror__(self, other): return Node.op('|',  other, self)
    def   __eq__(self, other): return Node.op('==',  self, other)
    def  __abs__(self): return Node.op('abs',  self)
    def   __lt__(self, other): return Node.op('<',  self, other)
    def   __gt__(self, other): return Node.op('>',  self, other)
    def   __le__(self, other): return Node.op('<=',  self, other)
    def   __ge__(self, other): return Node.op('>=',  self, other)
    def __lshift__(self, other): return Node.op('<<',  self, other)
    def __rshift__(self, other): return Node.op('>>',  self, other)
    def __mod__ (self, other): return Node.op('%',self, other)

    @staticmethod
    def sin(arg): return Node.op('sin', arg)
    @staticmethod
    def cos(arg): return Node.op('cos', arg)
    @staticmethod
    def if_then_else(cond, if_true, if_false=None):
        return Node.op('if_then_else', cond, if_true, if_false)

    #
    # Limited
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

    def limited(self):
        if type(self.value) is str:
            match self.value:
                case '+': return self[0].limited() + self[1].limited()
                case '-': return self[0].limited() - self[1].limited()
                case '*': return self[0].limited() * self[1].limited()
                case '/': return self[0].limited() / self[1].limited()
                case '**': return self[0].limited() ** self[1].limited()
                case '|': return self[0].limited() ** 0 ** self[1].limited()
                case '&': return self[0].limited() * self[1].limited()
                case 'neg': return 0 - self[0].limited()
                case '==': return 0 / (self[0].limited() - self[1].limited())
                case 'abs': return (self[0].limited() ** 2) ** (1 / 2)
                case '<':
                    s0 = self[0].limited()
                    s1 = self[1].limited()
                    return (1 - abs(s0 - s1) / (s0 - s1)) / 2
                case '>':
                    s0 = self[0].limited()
                    s1 = self[1].limited()
                    return (1 - abs(s1 - s0) / (s1 - s0)) / 2
                case '<=':
                    s0 = self[0].limited()
                    s1 = self[1].limited()
                    return (abs(s1 - s0) / (s1 - s0) + 1) / 2
                case '>=':
                    s0 = self[0].limited()
                    s1 = self[1].limited()
                    return (abs(s0 - s1) / (s1 - s0) + 1) / 2
                case '<<':
                    return self[0].limited() * 2 ** self[1].limited()
                case '>>':
                    s0 = self[0].limited()
                    s1 = self[1].value
                    if s1 == 0:
                        return s0
                    else:
                        s2 = (s0 >> s1-1) #.limited()
                        return ((s2 - s2 % 2) / 2).limited()
                case '%':
                    s0 = self[0].limited()
                    s1 = self[1].value
                    if s1 == 1:
                        return Node(0)
                    elif s1 == 2:
                        return (1 - (-1) ** s0) / 2
                    else:
                        k = int(math.log2(s1))
                        return ((((s0 >> k-1) % 2) << k-1) + (s0 % 2**(k-1))).limited()
                case _: return self
        else:
            return Node.const(self.value)

    # def  __neg__(self): return 0 - self
    # def  __and__(self, other): return self * other
    # def __rand__(self, other): return other * self
    # def   __or__(self, other): return self ** 0 ** other
    # def  __ror__(self, other): return other ** 0 ** self
    # def   __eq__(self, other): return 0 / (self - other)
    # def  __abs__(self): return (self * self) ** (1 / 2)
    # def   __lt__(self, other): return (1 - abs(self - other) / (self - other)) / 2
    # def   __gt__(self, other): return (1 - abs(other - self) / (other - self)) / 2
    # def   __le__(self, other): return (abs(other - self) / (other - self) + 1) / 2
    # def   __ge__(self, other): return (abs(self - other) / (self - other) + 1) / 2
    # def __lshift__(self, other): return self * 2 ** other
    # def __rshift__(self, other): return self if other == 0 else ((self >> other-1) - (self >> other-1) % 2) / 2

    # def __mod__(self, other):
    #     if other == 1:
    #         return Node(0)
    #     elif other == 2:
    #         return (1 - (-1) ** self) / 2
    #     else:
    #         k = int(math.log2(other))
    #         return (((self >> k-1) % 2) << k-1) + (self % 2**(k-1))
    #
    # @staticmethod
    # def min(*args): return args[0] * (args[0] < args[1]) + args[1] * (args[0] >= args[1])
    # @staticmethod
    # def max(*args): return args[0] * (args[0] > args[1]) + args[1] * (args[0] <= args[1])
    # @staticmethod
    # def if_then_else(cond, if_true, if_false=None):
    #     if if_false is None:
    #         return cond * if_true
    #     else:
    #         return cond * if_true + (1 - cond) * if_false
    # @staticmethod
    # def read_bit(n, x): return ((x % n) - (x % (n/2))) / (n/2)


if __name__ == '__main__':

    # x = Node('x')
    # y = Node('y')
    #
    # f0 = x + 1
    # f1 = f0 - x
    # f2 = f1 * f1
    # f3 = f2 / f1
    # f4 = f3 ** f2
    # f = f4

    # print(f(3))

    # f = f4.copy()
    # f0 = x + 1
    # f1 = f0 - f0
    # f2 = f1 * f1
    # f3 = f2 / f2
    # f4 = f3 ** f3
    # f = f4.copy()

    # f = x

    x = Node('x')

    f = (x + 1) + x

    # f0 = -x
    # f1 = Node.max(x, f0)
    # f2 = -f1
    # f3 = Node.cos(f1)
    # f4 = Node.cos(f2)
    # f5 = f3 - f4
    #
    # f = f5

    # f = x & x + 1
    # f = x % 4
    # f = x + 1

    print(f)
    print(f.limited())

    # plot_nodes([f, f.limited()], domains=[(0,31,32)])
    plot_tree(f)


    # ReLu
    # f = Node.if_then_else(x >= 0, x)
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
    # f = (((((x-x)+(x+x))**((x+x)/x))*(x+x))/((x+x)-x))
    # f = (((x+x)*((((((((x+x)*(x/x))*x)+(((x-(x*(x+x)))*x)+(x*x)))+((((x-((x*(((((x-x)-x)/x)+x)/x))/x))/x)*x)*x))+x)/((x-x)+x))-((x+x)*(0-x))))/((x+((x+(((((((((((x/((0/(x-x))*(((x+((x/x)-(x*x)))+(x+((x/x)*x)))*x)))+((x+x)/x))-(((((x/(x-x))+x)/(((((x+x)/x)-((x-x)+x))+x)/x))*x)*(x/x)))+x)+x)/x)-x)/(x*x))/x)-x)+x))/x))-x))
    # f = ((((x+x)**(((((x/x)+x)+x)+x)/x))-x)/((x+x)-x))
    # (((max((if_then_else(x,x,((x*(if_then_else((x|x),abs(x),x)|(abs(x)-(x+x))))|abs(x)))*x),abs(((((min(x,x)+((if_then_else(x,x,x)|(x-x))&x))+x)|x)+max((max((abs(((min(x,x)+max(((((x+min(x,x))|x)|x)+(x&x)),x))+(((((if_then_else(0,x,x)&(x/x))+min(if_then_else(x,x,x),min(x,x)))&(x+x))+abs(x))|(x|x))))+min(x,x)),x)+if_then_else((x-x),x,x)),x))))-x)-min(x,x))*x)
