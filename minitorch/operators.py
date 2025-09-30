"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.

def mul(a: float, b: float) -> float:
    return a*b
def id(a: float) -> float:
    return a
def add(a: float, b: float) -> float:
    return a + b
def neg(a: float) -> float:
    return float(-a)
def lt(a: float, b: float) -> bool:
    return a < b
def eq(a: float, b: float) -> bool:
    return a == b
def max(a: float, b: float) -> float:
    return a if a > b else b
def is_close(a: float, b: float) -> float:
    return abs(a - b) < 1e-2
def sigmoid(x: float) -> float:
    e = math.e
    return 1.0/(1.0 + e ** (-x)) if x >= 0 else e**x/(1.0 + e**x)
def relu(x: float) -> float:
    return x if x > 0.0 else 0.0
def log(x: float, base: float=math.e) -> float:
    return math.log(x, base)
def exp(x: float) -> float:
    return math.exp(x)
def log_back(x: float, y: float) -> float:
    return 1/x*y
def inv(x: float) -> float:
    return 1/x
def inv_back(a: float, b: float) -> float:
    return -1/(a**2) * b
def relu_back(x: float, y: float) -> float:
    return 0.0 if x < 0.0 else y

# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists

from typing import List

def map(f: Callable[[float], float], ls: Iterable[float]) -> Iterable[float]:
    for x in ls:
        yield f(x)

def zipWith(f: Callable[[float, float], float], ls: Iterable[float], rs: Iterable[float]) -> Iterable[float]:
    for x, y in zip(ls, rs):
        yield f(x, y)

from functools import reduce as snatched_reduce
def reduce(f: Callable[[float, float], float], ls: List[float]) -> float:
    if not ls:
        return 0.0
    return snatched_reduce(f, ls)

def negList(ls: List[float]):
    return map(neg, ls)

def addLists(ls: List[float], rs: List[float]):
    return zipWith(add, ls, rs)

def sum(ls: List[float]) -> float:
    return reduce(add, ls)

def prod(ls: List[float]) -> float:
    return reduce(mul, ls)

# new
def sigmoid_back(x: float, y: float) -> float:
    return sigmoid(x) * (1 - sigmoid(x)) * y

def exp_back(x: float, y: float) -> float:
    return exp(x) * y
