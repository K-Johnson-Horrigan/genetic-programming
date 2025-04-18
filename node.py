import itertools
import math
from inspect import signature
# from np import sin, cos

import numpy as np
import sympy as sp
from numpy.ma.core import ones_like
from numpy.random import permutation

from plot import plot_effective, plot_graph


class Node:
    """A basic class for genetic programming. A Node holds a single value and points to zero or more children Nodes."""

    # All possible values for a node and the number of children it can have
    valid_ops = {
        'noop': 1,
        'neg': 1,
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

    def __init__(self, value, children=None):
        self.parent = None
        self.parents = []
        # If the value is already a node use its value so that Nodes can be cast to a Node
        # This also allows for copies of a Node to be made through casting
        if type(value) == Node:
            self.children = value.copy().children
            self.value = value.value
        else:
            self.value = value
            self.children = children if children is not None else []
        # Used when creating a list of all nodes to prevent repeats.
        # None indicates that all children also have a temp_index of None
        # Setting this to -1 and then resetting results in it being None
        self.temp_index = -1
        # Previously returned value used for semantic analysis
        self.returned_value = -1
        # If all descendants are in the simplest form
        self.is_limited = False

    #
    # Children and Parents
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

    def reset_parents(self):
        """Remove all parent pointers from all nodes"""
        if len(self.parents) > 0:
            self.parents = []
            for child in self:
                child.reset_parents()

    def set_parents(self):
        """Append all parent pointers back to the nodes"""
        for child in self.children:
            child.parents.append(self)
            child.set_parents()

    #
    # Utility
    #

    def reset_returned_value(self):
        """Set the returned_value of all nodes to None"""
        if self.returned_value is not None:
            self.returned_value = None
            for child in self.children:
                child.reset_index()

    def reset_index(self):
        """Set the temp_index of all nodes to None"""
        if self.temp_index is not None:
            self.temp_index = None
            for child in self.children:
                child.reset_index()

    def index_in(self, l):
        """Returns the first index of this object in the given iterable. The `in` keyword and `index` method will not work for Nodes"""
        for i,node in enumerate(l):
            if node is self:
                return i
        return -1

    def copy(self):
        return Node.from_lists(*self.to_lists())

    #
    # Information
    #

    def height(self):
        """Longest distance to a leaf"""
        return max([0] + [1 + child.height() for child in self.children])

    def depth(self):
        """Longest distance to the root"""
        return max([0] + [1 + parent.depth() for parent in self.parents])

    def root(self):
        """Returns the root Node of the tree"""
        return self if self.parent is None else self.parent.root()

    def size(self):
        """Returns the number of nodes"""
        return len(self.nodes())

    # def num_neutral(self):
    #     """Count the operations that always return the same operand (identity or absorbing)"""
    #     if type(self.value) is str:
    #         match self.value:
    #             case 'noop':
    #                 return 1 + sum([c.num_neutral() for c in self])
    #             case '+':
    #                 s0 = self[1].simplify()
    #                 s1 = self[0].simplify()
    #                 if s0 == 0 or s1 == 0:
    #                     return 1 + self[0].num_neutral() + self[1].num_neutral()
    #             case '-':
    #                 s1 = self[0].simplify()
    #                 if s1 == 0:
    #                     return 1 + self[0].num_neutral() + self[1].num_neutral()
    #             case '*':
    #                 s0 = self[1].simplify()
    #                 s1 = self[0].simplify()
    #                 if s0 == 0 or s1 == 0 or s0 == 1 or s1 == 1:
    #                     return 1 + self[0].num_neutral() + self[1].num_neutral()
    #             case '/':
    #                 s0 = self[1].simplify()
    #                 s1 = self[0].simplify()
    #                 if s0 == 0 or s1 == 1 or (s0 == 1 and s1 == 0):
    #                     return 1 + self[0].num_neutral() + self[1].num_neutral()
    #             case '**':
    #                 s0 = self[1].simplify()
    #                 s1 = self[0].simplify()
    #                 if (s1 == 1) or (s0 == 0 and s1 != 0) or (s0 == 1 and s1 == 0):
    #                     return 1 + self[0].num_neutral() + self[1].num_neutral()
    #     return sum([c.num_neutral() for c in self])

    def effective_code(self, a=None):
        """The effective code of the last evaluation"""
        init_call = a is None
        a = [] if init_call else a
        # Call recursively for each child
        for child in self:
            child.effective_code(a)
            a.append(np.linalg.norm(self.returned_value - child.returned_value))
        # Calculate the result from all semantic vectors
        if init_call:
            effective_code_value = np.sum(np.bool(a)) / (len(self.nodes()) - 1)
            effective_code_value = np.nan_to_num(effective_code_value, nan=0)
            return effective_code_value

    #
    # String Representation
    #

    def __str__(self):
        if len(self) == 0:
            return str(self.value)
        # elif self.value in ['+','-','*','/','**','&','|','%','>>','<<','<','>','<=','>=','==']:
        elif len(self.value) <= 2:
            return f'({self[0]}{self.value}{self[1]})'
        else:
            return self.value + '(' + ','.join([str(child) for child in self]) + ')'

    def __repr__(self):
        return str(self)

    def latex(self):
        return sp.latex(self.simplify())

    #
    # Modification
    #

    def replace(self, new_node):
        """Replaces this node and all children with a new branch"""
        root = self.root()
        # Create a copy of the new node
        # new_node = new_node.copy()
        new_node = new_node
        # Return the new node if self is the root of the tree
        if len(self.parents) == 0:
            self.value = new_node.value
            self.children = new_node.children
        # Change all nodes pointing at this node to be pointing at the new node
        for parent in self.parents:
            # Parent's index for self
            self_index = self.index_in(parent)
            # Replace the parent's reference to self
            parent[self_index] = new_node
        # Recalculate all links to parents
        # This is because the original structure may still point to descendants of the original
        root.reset_parents()
        root.set_parents()
        # Return the full new tree
        return root

    #
    # Conversion
    #

    def to_tree(self):
        """Returns a recursive deepcopy of all Nodes"""
        return Node(self.value, [child.to_tree() for child in self])

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

    #
    # Evaluation
    #

    def __call__(self, *x, eval_method=None, **kwargs):
        """Calling evaluates the value of the entire tree. Values of x can be numbers or ndarrays"""

        match eval_method:

            # Returns x once f(x)==0 otherwise x:=f(x)
            case 'zero':
                return_value = self(*x)
                for _ in range(100):
                    new_return_value = self(return_value)
                    print(return_value)
                    if new_return_value == 0:
                        return new_return_value
                    return_value = new_return_value
                return return_value

            # Evaluate x:=f(x) until even
            case 'even':
                return_value = self(*x)
                for _ in range(100):
                    return_value = self(return_value)
                    if return_value % 2 == 0: break
                return return_value // 2

            # Default evaluation
            case _:
                if type(self.value) is not str:
                    # A node with a number value should return in the shape of the
                    return_value = self.value * np.ones_like(x[0])
                else:
                    match self.value:
                        # Operations
                        case '+': return_value = self[0](*x, **kwargs) + self[1](*x, **kwargs)
                        case '-':
                            return_value = self[0](*x, **kwargs) - self[1](*x, **kwargs)

                        case '*':
                            # s0 = self[0](*x, **kwargs)
                            # if s0 == 0:
                            #     return_value = 0
                            # else:
                            #     return_value = s0 * self[1](*x, **kwargs)
                            return_value = self[0](*x, **kwargs) * self[1](*x, **kwargs)

                        case '/':
                            s0, s1 = self[0](*x, **kwargs), self[1](*x, **kwargs)
                            # return_value = 1 if s1 == 0 else s0 / s1

                            return_value = ones_like(s0, 'complex')
                            ind = s1 != 0
                            np.true_divide(s0, s1, out=return_value, where=ind, dtype='complex')
                            return_value = np.array(return_value)
                            # return_value = complex(return_value)

                        case '**':
                            # for ind in range(x.shape):
                            s0, s1 = self[0](*x, **kwargs), self[1](*x, **kwargs)

                            # if s0 == 0 and (np.isreal(s1) or np.real(s1) < 0):
                            #     return_value = 1
                            # else:
                                # Prevent large exponents using numpy
                                # return_value = np.power(1.0*s0, s1)

                            return_value = ones_like(s0, 'complex')
                            # Valid where s0 is not zero or s1 is a positive real number
                            ind = (s0 != 0) | (np.isreal(s1) & (np.real(s1) > 0))
                            # print(ind)
                            # return_value[ind] = np.power(s0[ind], s1[ind], where=ind)
                            np.power(s0, s1, out=return_value, where=ind, dtype='complex')

                        case 'noop': return_value = self[0](*x, **kwargs)
                        case 'neg': return_value = -self[0](*x, **kwargs)
                        case '|': return_value = self[0](*x, **kwargs) | self[1](*x, **kwargs)
                        case '&': return_value = self[0](*x, **kwargs) & self[1](*x, **kwargs)
                        case '<': return_value = self[0](*x, **kwargs) < self[1](*x, **kwargs)
                        case '>': return_value = self[0](*x, **kwargs) > self[1](*x, **kwargs)
                        case '<=': return_value = self[0](*x, **kwargs) <= self[1](*x, **kwargs)
                        case '>=': return_value = self[0](*x, **kwargs) >= self[1](*x, **kwargs)
                        case '==': return_value = self[0](*x, **kwargs) == self[1](*x, **kwargs)
                        case 'min': return_value = min(self[0](*x, **kwargs), self[1](*x, **kwargs))
                        case 'max': return_value = max(self[0](*x, **kwargs), self[1](*x, **kwargs))
                        case 'abs': return_value = abs(self[0](*x, **kwargs))
                        case 'if_then_else': return_value = self[1](*x, **kwargs) if self[0](*x, **kwargs) else self[2](*x, **kwargs)
                        case '%':  return_value = self[0](*x, **kwargs) % self[1](*x, **kwargs)
                        case '>>': return_value = self[0](*x, **kwargs) >> self[1](*x, **kwargs)
                        case '<<': return_value = self[0](*x, **kwargs) << self[1](*x, **kwargs)
                        case 'sin': return_value = np.sin(self[0](*x, **kwargs))
                        case 'cos': return_value = np.cos(self[0](*x, **kwargs))
                        case 'get_bit': return_value = (int(self[0](*x, **kwargs)) >> self[1](*x, **kwargs)) & 1
                        case 'get_bits':
                            s0, s1, s2 = self[0](*x, **kwargs), self[1](*x, **kwargs), self[2](*x, **kwargs)
                            return_value = (int(s0) >> s1) % (2 ** s2)

                        # Terminals and constants
                        case 'x': return_value = x[0]
                        case 'y': return_value = x[1]
                        case 'z': return_value = x[2]
                        case 'e': return_value = kwargs['e'] if 'e' in kwargs else np.e * np.ones_like(x[0])
                        case 'i': return_value = kwargs['i'] if 'i' in kwargs else 1j * np.ones_like(x[0])

                        # Arbitrary Variable
                        case _: return_value = np.float64(x[int(''.join([s for s in self.value if s.isdigit()]))])

                # Store the last returned value for analysis if needed
                self.returned_value = return_value
                return return_value

    def simplify(self):
        return sp.sympify(self(sp.Symbol('x'), e=sp.E, i=sp.I))

    #
    # Construction From Native Python Operations
    #

    @staticmethod
    def op(operation, *operands):
        """Return a new Node from an operation on other Nodes"""
        # Convert operands to a list to be modified
        operands = list(operands)
        # Cast each operand to a Node, operands must not be copied as pointers need to be preserved
        for i in range(len(operands)):
            if type(operands[i]) != Node:
                operands[i] = Node(operands[i])
        # Return a new Node with the operands as the children
        new_node = Node(operation, operands)
        # Maintain that the root node has the same attributes TODO: improve implementation
        if hasattr(operands[0], 'prev_fit'):
            new_node.prev_fit = operands[0].prev_fit
        return new_node

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
    def      __neg__(self       ): return Node.op('neg',self       )
    def      __and__(self, other): return Node.op('&',  self, other)
    def     __rand__(self, other): return Node.op('&',  other, self)
    def       __or__(self, other): return Node.op('|',  self, other)
    def      __ror__(self, other): return Node.op('|',  other, self)
    def       __eq__(self, other): return Node.op('==', self, other)
    def      __abs__(self       ): return Node.op('abs',self       )
    def       __lt__(self, other): return Node.op('<',  self, other)
    def       __gt__(self, other): return Node.op('>',  self, other)
    def       __le__(self, other): return Node.op('<=', self, other)
    def       __ge__(self, other): return Node.op('>=', self, other)
    def   __lshift__(self, other): return Node.op('<<', self, other)
    def   __rshift__(self, other): return Node.op('>>', self, other)
    def      __mod__(self, other): return Node.op('%',  self, other)

    @staticmethod
    def max(*args): return Node.op('max', *args)
    @staticmethod
    def min(*args): return Node.op('min', *args)
    @staticmethod
    def sin(arg): return Node.op('sin', arg)
    @staticmethod
    def cos(arg): return Node.op('cos', arg)
    @staticmethod
    def get_bits(f, start, length): return Node.op('get_bits', f, start, length)
    @staticmethod
    def if_then_else(cond, if_true, if_false): return Node.op('if_then_else', cond, if_true, if_false)

    #
    # Limited Equivalence
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
        if self.is_limited:
            return self
        elif type(self.value) is not str:
            return_value = Node.const(self.value)
        else:
            match self.value:
                case '+' | '-' | '*' | '/' | '**':
                    self.children = [child.limited() for child in self]
                    self.is_limited = True
                    return self
                case 'neg':
                    return_value = 0 - self[0]
                case '|':
                    return_value = self[0] ** 0 ** self[1]
                case '&':
                    return_value = self[0] * self[1]
                case '==':
                    return_value = 0 / (self[0] - self[1])
                case 'abs':
                    return_value = (self[0] * self[0]) ** (Node.const(1) / 2)
                case '<':

                    return_value = (1 - abs(self[0] - self[1]) / (self[0] - self[1])) / 2
                case '>':
                    return_value = (1 - abs(self[1] - self[0]) / (self[1] - self[0])) / 2
                case '<=':
                    return_value = ((abs(self[1] - self[0]) / (self[1] - self[0]) + 1) / 2)
                case '>=':
                    return_value = ((abs(self[0] - self[1]) / (self[1] - self[0]) + 1) / 2)
                case '<<':
                    return_value = (self[0] * 2 ** self[1]).limited()
                case '>>':
                    s1 = self[1].value
                    if s1 == 0:
                        return_value = self[0]
                    else:
                        rec = (self[0] >> s1-1)
                        return_value = ((rec - rec % 2) / 2)
                case '%':
                    if self[1].value == 1:
                        return_value = Node(0)
                    elif self[1].value == 2:
                        return_value = (1 - (-1) ** self[0]) / 2
                    else:
                        k = int(math.log2(self[1].value))
                        return_value = ((((self[0] >> k-1) % 2) << k-1) + (self[0] % 2**(k-1)))
                case 'sin':
                    e = Node('e')
                    i = Node('i')
                    return_value = ((e ** (i * self[0]) - e ** (-i * self[0])) / (2 * i)).limited()
                case 'cos':
                    e = Node('e')
                    i = Node('i')
                    return_value = ((e ** (i * self[0]) + e ** (-i * self[0])) / 2).limited()
                case 'get_bits':
                    return_value = ((self[0] >> self[1]) % (2 ** self[2])).limited()
                case _: return self

        return_value = return_value.limited()
        self.replace(return_value)
        return_value.is_limited = True
        return return_value




x = Node('x')
y = Node('y')
z = Node('z')
i = Node('i')
e = Node('e')

if __name__ == '__main__':

    f = (x-y)/x

    # f = x ** x

    print(f)
    print(f(0,1))

    # j = -y

    # f = j
    # f = j * j * j
    # f = f < 3
    # j.replace(z)
    # f = f.limited()

    # f0 = x + 1
    # f1 = x - f0
    # f2 = f0 * f1
    #
    # f = f2



    # f1.replace(z)
    # f = Node.sin(x).limited()

    # plot_graph(f)

    # print(f(0, 0))
    # print(f(0., 0j))
    # # print(f(1, 0))
    # print(f(np.array([0,0,1,1]), np.array([0,1,0,1])))


    # f = Node.if_then_else(
    #     x == 1,
    #     0,
    #     Node.if_then_else(
    #         x % 2,
    #         3 * x + 1,
    #         x / 2,
    #     )
    # )

    # l = f.limited()

    # f1 = x + 1
    # g = f1.copy()
    # f2 = f1 - f1
    # # f = g
    # f = f2

    # f = 0 * x * x * x * x
    # f = x * x * x * x * 0

    # print(f.num_neutral())

    # f = (x + 1) + 2
    # f = (x + 1)
    # xs = np.array([2,3])
    # # xs = 1
    # print(f(xs))
    # ec = f.effective_code()
    # print(ec)
    # print(np.sum(ec, axis=0))

    # all_pops = np.empty((1,1,1,1))
    # all_pops[:] = f

    # plot_effective(all_pops)

    # f = (x % 4).limited()

    # print(f.to_lists())
    # plot_graph(f)

    # print(f(4, eval_method='zero'))
    #
    # plot_nodes(
    #     [f],
    #     domains=((0,15,16),),
    #     eval_method='zero'
    # )
    # plot_graph(f.limited(), 1)

    # print(l)

    # print(l.simplify())

    # f = x + x
    #
    # l = [x, 0, x, f]
    #
    # i = f.index_in(l)
    #
    # x = Node('x')
    # y = Node('y')
    # r = [Node(str(i)) for i in range(6)]
    #
    #
    # a = (r[4] - r[3]) + (r[1] + r[5])
    # b = (r[1] / r[2]) * (r[3] + r[1])
    #
    # a0 =  x + 0
    # a1 = a0 + x
    # a2 = a1 + a0
    # a = a2
    #
    # print(a)
    # plot_tree(a)
    # a1.replace(y)
    # plot_tree(a)
    # print(a)

    # print(i)

    # g = x**5 + 32*x**3 + x

    # f = ((-21.077945511687545 * (x - ((x * x) * (x + x)))) + -1.218498355689607e-07)

    # f = ((64.33472816266729*((((x*x)*x)*x)/(((x+x)-x)*(x+x))))+-0.08907341519219569)
    #
    # print((f).simplify())


    # plot_nodes([f, f.limited()], domains=[(0, 2 * math.pi, 16)])
    # f = (x_0 / (x_1 - ((x_1 / x_0) - x_0)))
    # print(f(1,1))

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

    # f = x + x
    # f = f + f
    # f = f + f.expanded_copy()
    # f = f.expanded_copy()

    # f = x

    # f = -x

    # f0 = -x
    # f1 = Node.max(x, f0)
    # f2 = -f1
    # f3 = Node.cos(f1)
    # f4 = Node.cos(f2)
    # f5 = f3 - f4
    # f = f5

    # f = x & x + 1
    # f = x % 4
    # f = x + 1

    # print(f)
    # print(f.limited())
    # print(f.limited().to_lists())

    # plot_nodes([f, f.limited()], domains=[(0,31,32)])
    # plot_tree(f.limited(), 0)


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
    # s = 8
    # n = Node.get_bits(x, 0, s)
    # c = Node.get_bits(x, s, s) + 1
    # # c=c n=n
    # # n == 1
    # # c=0 n=c
    # # c == 0
    # # c=0 n=0
    # cc = Node.if_then_else(
    #     n == 1,
    #     -1,
    #     Node.if_then_else(
    #         c == 0,
    #         0,
    #         c
    #     ),
    # )
    # nn = Node.if_then_else(
    #     n == 1,
    #     c,
    #     Node.if_then_else(
    #         c == 0,
    #         0,
    #         Node.if_then_else(
    #             n % 2,
    #             3 * n + 1,
    #             n / 2,
    #         )
    #     )
    # )
    # f = cc * 2**s + nn
