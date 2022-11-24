"""Method Resolution Order (MRO) is the order in which Python looks
for a method in a hierarchy of classes. Especially it plays vital role
in the context of multiple inheritance as single method may be found
in multiple super classes.
"""


class Type(type):
    def __repr__(cls):
        return cls.__name__


class O(object, metaclass=Type): pass


class E(O): pass


class D(O): pass


class F(O): pass


class B(E, D): pass


class C(D, F): pass


class A(C, B): pass




"""
RESULT FOR print(A.mro())
# [A, B, C, D, E, F, O, <class 'object'>]

EXPLANATION: 
From MRO of class A, python looks for methods in class A first, then to the super class given first in the class,
i.e B then to the second, etc, and finally looks at the Object class.
The MRO checks from left to right, hence, upon a conflict, simply interchanging the classes 
turns out to be the solution.
"""

# Complexity of Hierarchy Scenario


class Aa:
    def process(self):
        print('A process()')


class Bb(Aa):
    def process(self):
        print('B process()')


class Cc(Aa, Bb):
    pass


obj = Cc()
obj.process()


if __name__ == "__main__":
    # print(A.mro())
    # print(B.mro())
    # print(C.mro())
    obj = Cc()
    print(obj.process())