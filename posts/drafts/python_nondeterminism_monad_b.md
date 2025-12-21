## Scaling it up

This is certainly interesting, but even ignoring the efficiency loss from having to reconstruct the entire function state (by rewinding the generator) each time we follow a branch, the performance leaves much to be desired: Python is not a fast language. To illustrate, we'll try implementing a naive function that generates a list of [Pythagorean triples](https://en.wikipedia.org/wiki/Pythagorean_triple) with a, b, c in `1..200`:

```py
guard = lambda c: Amb([None]) if c else Amb([])

@amb
def pythagorean_triples(n: int) -> set[Tuple[int, int, int]]:
    x = yield Amb(range(1, n+1))
    y = yield Amb(range(x+1, n+1)) # avoid double-counting
    z = yield Amb(range(y+1, n+1))
    yield guard(x ** 2 + y ** 2 == z ** 2)
    return (x, y, z)

print(len(t := pythagorean_triples(200)))
pprint(t)
```

`time python amb.py` gives:

```
127
{(3, 4, 5),
 (5, 12, 13),
 (6, 8, 10),
 (7, 24, 25),
 (8, 15, 17),
 (9, 12, 15),
 (9, 40, 41),
 (10, 24, 26),
 (11, 60, 61),
 (12, 16, 20),
 (12, 35, 37),
 (13, 84, 85),
 (14, 48, 50),
 (15, 20, 25),

 ...

 python amb.py  6.88s user 0.02s system 99% cpu 6.932 total
```

### Vectorized combinatorial sugar

This is somewhat better than I expected, but still impractical for most real-world problems. For performing e.g., Monte Carlo simulations, we want something with performance within an order of magnitude of C/C++ code (or at least Haskell). The ideal case would be to somehow vectorize the annotated function with minimal input from the user, probably using NumPy or a similar library that provides Python bindings to efficient array operations. In particular, if each step (or bind operation) in our function can be represented by `f(a, b, ...)`, with each argument being some (nondeterministic) expression derived from an `Amb` term, we want to implicitly generate the Cartesian product of all `a_i, b_j, ...` and pass it to a vectorized version of `f`. This will inevitably require some syntactic and semantic tradeoffs over the generator-based version. For now, we'll impose the constraint that our decorated function only takes as inputs, computes using, or returns integers or floats.

Unfortunately, just swapping NumPy arrays into the code we showed earlier (instead of lists/sets) probably wouldn't be of much use: we'd still end up iterating through every element in native Python. We would inevitably need to have a vectorized version of every operation involved in the code so that we could process many branches in parallel. The other extreme involves full vectorization ignoring control flow; after each `Amb`, we could unroll all that had been encountered up to that point, take their Cartesian product, and compute every relevant operation on every combination of inputs, accepting some wasted work in exchange for being able to forego extremely expensive native Python logic. In the case of our Pythagorean triple calculator, we don't lose much, since we need to consider every combination of elements from our three `Amb` expressions (clearly, a simpler collection-based solution like some of the ones shown on Rosetta Code would also work for this problem).

If we are willing to dispense for the moment with some more complex control flow, we may as well just replace the generator-based interceptor with a special object that broadcasts over common arithmetic operations; if we do something like `Wrapped([1, 2]) + Wrapped([3, 4])`, for example, we would expect to get `Wrapped([1 + 3, 2 + 3, 1 + 4, 2 + 4]) == Wrapped([4, 5, 5, 6])`. This is certainly cleaner than with the generator approach; we only traverse the code once, instead of once per branch/combination. The main trouble is with if-statements -- we must follow *both* branches at different places in the array, ideally rewriting the `if` as an `np.where` or similar (we can perform this translation manually, but figuring out how to avoid this is an interesting exercise).

Let's create a wrapper class that stands in for scalar values in our program and automatically performs the broadcasting described above:

```py
import numpy as np
import operator

class Amb2:
    def __init__(self, data):
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        self.data = data

def makeop(name: str):
    def tmp(xs: Amb2, ys: Amb2) -> Amb2:
        a, b = np.meshgrid(xs.data, ys.data, indexing='ij')
        prod = np.stack([a.ravel(), b.ravel()], axis=-1)
        return Amb2(getattr(operator, name)(prod[:, 0], prod[:, 1]))

    setattr(Amb2, f'__{name}__', tmp)

for f in ['add', 'mul', 'sub', 'truediv', 'floordiv', 'pow', 'and_', 'or_', 'xor']:
    makeop(f)

def amb(f):
    def r(*args, **kwargs):
        return f(*args, **kwargs).data
    return r

@amb
def test() -> np.ndarray:
    a = Amb2([1, 3, 5])
    b = Amb2([2, 4, 6])
    return a + b

print(test())
```

```
[ 3  5  7  5  7  9  7  9 11]
```

It's probably also worthwhile to support scalar types mixed into our code without additional effort from the programmer:

```py
def makeop(name: str):
    def tmp(xs: Amb2 | int | float, ys: Amb2 | int | float) -> Amb2:
        if not isinstance(xs, Amb2):
            assert isinstance(xs, (int, float))
            xs = Amb2([xs])
        if not isinstance(ys, Amb2):
            assert isinstance(ys, (int, float))
            ys = Amb2([ys])

        a, b = np.meshgrid(xs.data, ys.data, indexing='ij')
        prod = np.stack([a.ravel(), b.ravel()], axis=-1)
        return Amb2(getattr(operator, name)(prod[:, 0], prod[:, 1]))

    setattr(Amb2, f'__{name}__', tmp)
    setattr(Amb2, f'__r{name}__', tmp)

...

@amb
def test() -> np.ndarray:
    a = Amb2([1, 3, 5])
    b = Amb2([2, 4, 6])
    return 1.5 * (a + b + 2)

print(test())
```

```
[ 7.5 10.5 13.5 10.5 13.5 16.5 13.5 16.5 19.5]
```

Mutation is also supported, even if the left-hand side is not (yet) `Amb`:

```py
@amb
def test2() -> np.ndarray:
    a = 1
    for i in range(3):
        a += Amb2([2, 3])
    return a
```

```
[ 7  8  8  9  8  9  9 10]
```

For/while loops (and conditionals) however only work when the loop condition is "primitive" and does not contain any `Amb`-expressions. For example, a while-loop with a condition derived from an `Amb` would in general not behave as expected; one could imagine a way to make this work by detecting when we drop into a loop, overriding the behavior of `bool`-coercion to keep the loop running until the condition is false for *all* values in the `Amb`, and "masking" assignment operations so that they only affect members of the target value for which corresponding values of any ambiguous expressions in the context still make the loop condition evaluate to `True` (perhaps maintaining a stack of contexts for nested loops/conditional statements), all in an efficient vectorized fashion. This is nontrivial.

The main issue here is that we discard provenance information describing where the values were derived from; we could cache the expression tree used to generate the concrete values for each `Amb`, but this wouldn't be of much use if we wanted to e.g., perform another computation using the same variables and use it to filter the results of the original computation.

What we would really like to do is assign a unique array dimension to each primitive `Amb`-value (recall, these represent the sets that our ambiguous/nondeterministic variables vary over), such that e.g., a binary operation between two such expressions (conceptually) broadcasts one against the other, producing a 2D array containing the output of the operator for every possible pair of input values (i.e., a Cartesian product). If we shape our (input and output) arrays such that they have non-1 width only on exactly those axes corresponding to the `Amb`s used in the expression. To illustrate, the product of three arrays with shapes `(4, 1, 1, 1)`, `(1, 5, 1, 1)`, and `(1, 1, 1, 6)` will have shape `(4, 5, 1, 6)`, regardless of the order in which we do the multiplications (`(a * b) * c` or `a * (b * c)`). In this case, we may have three `Amb`s with size 4, 5, and 6 assigned axes 0, 1, and 3, respectively.

For very sparse computations or ones that naturally admit some amount of branching/pruning early on, this will obviously waste a lot of computation, but in practice the speedup from running vectorized array code is often so overwhelming that this doesn't matter (and it's not too hard to imagine other optimizations that cut down on the overhead -- though many of these require a more global view of the data graph, so our approach would need some restructuring). For this reason it is not tremendously useful to include a "range" operator that can be used to express, e.g., looping over every integer between the value of some `Amb`-expression and a constant upper bound; the natural way to TODO. You may see the parallel between this and the GPU computation model, in which we must generally execute *both* branches of a conditional since we cannot efficiently appraise ahead of time which values the branching condition will be true for, and the hardware specifically targets highly parallel workloads that can be processed with relatively little branching.

[TODO: rewrite this, it was reused for the end of part 1] I may address some of this in more detail in a future post, but for now I've opted to end it here in the interest of being able to get this out in a timely manner. Hopefully I've provided enough context to get you thinking about how some of the aforementioned issues might be addressed (particularly, how you might use runtime execution tracing, potentially augmented with cpython's code introspection facilities, to handle unadorned loops and conditionals).
