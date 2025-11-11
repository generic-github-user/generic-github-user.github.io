# Nondeterminism monad in Python

**Also in this post: Haskell and parsing-free execution tracers**

*This post contains some fairly lengthy exposition; if you want to skip right to the content promised by the title, go to `## Nondeterminism in Python`*

Many of the most curious and useful features of functional programming languages like Haskell derive from their ability (often unencumbered by the norms and constraints of industrial software engineering) to restate common algorithmic problems in novel ways -- i.e., to perform a change of basis into a domain more suited to the problem. One such frame shift (or rather, category of such) is widely known as [*declarative programming*](https://en.wikipedia.org/wiki/Declarative_programming) (as opposed to imperative or functional programming, for example), and concerns programming languages, libraries, and techniques based on stating the problem domain or constraint system, as well as the desired objective or target (the "what"), at a high level and leaving the low-level algorithmic details to the optimizer or runtime (the "how"). In some cases this may take the form of a domain-specific optimization or constraint solving library; other times it is integrated more tightly with a language's semantics and execution model.

One self-contained and useful tool from this paradigm is "nondeterminism" (this is a somewhat overloaded term, but in this case I am *not* talking about the kind of nondeterminism that people mention with respect to e.g., reproducibility of software artifacts or experiments). The premise is that we delineate the ways in which a program can branch or the alternatives to be selected between (potentially with nesting, recursion, and other complications) and search/"solve" for some solution among the set of possible paths or choices. That is to say, the nondeterminism interface should abstract away the construction of the search space and execution of the search procedure to some extent; the programmer need only be concerned with which choices are available and how they interact (e.g., how state evolves over time depending on the branch taken at each step).

The classical implementation of this scheme is described in chapter 4.3 of the well-known [*Structure and Interpretation of Computer Programs*](https://en.wikipedia.org/wiki/Structure_and_Interpretation_of_Computer_Programs): McCarthy's `amb` ("ambiguous") operator, usually implemented in Lisp. I will omit the details of the implementation since there are [decent](https://rosettacode.org/wiki/Amb) [explanations](https://www.randomhacks.net/2005/10/11/amb-operator/) [elsewhere](https://ds26gte.github.io/tyscheme/index-Z-H-16.html), but the idea rhymes with what I've described above: we consider "splitting" execution into different paths corresponding to available combinations of initial values, and return results only from the path (or paths) where execution "succeeded" (some specified criteria was met).

## Nondeterminism in the list monad

[Haskell](https://www.haskell.org/) has what I would consider a somewhat more principled implementation of nondeterminism using monads. In particular, the built-in list type forms a monad, with `\xs f -> concat (map f xs)` as the `>>=` (`bind`) operation (and the singleton list constructor, i.e., `return x = [x]`, as `return`/`pure`). This means that the resulting list will be constructed by passing each element in `xs` to `f` to yield a new list, then concatenating the results:

```hs
ghci> [1, 2, 3] >>= (\x -> [x * 2, x * 3])
[2,3,4,6,6,9]
```

(see [this Wikibooks entry](https://en.wikibooks.org/wiki/Haskell/Understanding_monads/List) for a more detailed explanation)

As you may expect, this means that we can stack multiple "branching" operations to recursively expand every possible path:

```hs
ghci> [1, 2, 3] >>= (\x -> [x * 2, x * 3]) >>= (\y -> [y + 4, 0])
[6,0,7,0,8,0,10,0,10,0,13,0]
```

The equivalent code in [do-notation](https://en.wikibooks.org/wiki/Haskell/do_notation) looks like this:

```hs
xs = do
	x <- [1, 2, 3]
	y <- [x * 2, x * 3]
	z <- [y + 4, 0]
	return z
```

Perhaps now it is clear how this is useful: we can trivially iterate over all combinations of choices for x, y, and z merely by specifying what the choices are, foregoing cumbersome and unscalable combinatorics logic. If you're familiar with Haskell's list comprehension notation, this is indeed syntax sugar for the list monad:

```hs
ghci> [((x, y), x * y) | x <- [1, 2, 3], y <- [4, 5, 6]]
[((1,4),4),((1,5),5),((1,6),6),((2,4),8),((2,5),10),((2,6),12),((3,4),12),((3,5),15),((3,6),18)]

ghci> [1, 2, 3] >>= (\x -> [4, 5, 6] >>= (\y -> [((x, y), x * y)]))
[((1,4),4),((1,5),5),((1,6),6),((2,4),8),((2,5),10),((2,6),12),((3,4),12),((3,5),15),((3,6),18)]
```

We can do more interesting things as well; `Control.Monad` exports a function called `guard` with the following (quite general) definition:

```hs
guard True  = pure ()
guard False = empty
```

...which for `[]`, specializes to:

```hs
ghci> (guard True) :: [()]
[()]
ghci> (guard False) :: [()]
[]
```

`guard` lets us access a general "cancellation" action for applicative functors (specifically, `Alternative`s); in the context of the list monad, we can think of this as conditionally pruning a branch from our computation by ignoring the results accumulated so far in that branch and returning an empty list. Let's say we want to find all pairs of integers in `1..10 x 1..10` with even products, and annotate them with those products:

```hs
import Control.Monad

xs = do
  a <- [1..10]
  b <- [1..10]
  guard (even (a * b))
  return ((a, b), a * b)

main = mapM print xs
```

```
((1,2),2)
((1,4),4)
((1,6),6)
((1,8),8)
((1,10),10)
((2,1),2)
((2,2),4)
((2,3),6)
((2,4),8)
((2,5),10)
((2,6),12)
...
```


Haskell has convenient syntax sugar for this too:

```
ghci> mapM print [((x, y), x * y) | x <- [1..10], y <- [1..10], even (x * y)]
((1,2),2)
((1,4),4)
((1,6),6)
((1,8),8)
((1,10),10)
((2,1),2)
```

(There is a tremendous amount of fascinating monad/applicative/traversable/alternative machinery that works with almost all of Haskell's basic types, and which I would recommend having a look at if the above interests you at all; another example I'm fond of is `sequence [[1..2], [3..6]]`, which exploits the fact that lists are both `Traversable` and `Monad`.)

It is important to note that -- unlike with naive implementations that iterate over all combinations of elements from a handful of statically known sets, and only check which combinations would have survived at the end -- this approach really does prune branches each time `guard` is invoked, avoiding much unnecessary work, and has all the execution semantics you would expect of a handwritten "iterate over items in source collections -> map transformations over each element and collect the results -> filter/prune -> ..." approach.

(The main deficiency, aside from those mild to moderate performance concerns (cache locality, laziness) that apply to Haskell's execution model more generally, is that since `Data.Set` cannot be made into a monad (for any `Set a`, `a` carries an `Ord` constraint), we cannot use the obvious optimization strategy: "implement nondeterminism backed using a `Set` so that at each step/branching point, the universe is automatically collapsed down into unique values". There are some packages which claim to implement a performant, monad-compatible set datatype, the implementation details of which I know not.)

Another very compelling feature of Haskell, which is a bit of a diversion from the main subject of this post but still worth bringing up, is that monad transformers can be used to mix and match nondeterminism with other kinds of effects -- for example, the early termination/short-circuiting behavior of the `Maybe` monad, or the hermetic state manipulation features of `State`. As a brief example, we can use `StateT` over the list monad to iterate over all combinations of some transformations (successively applied to an initial value) while maintaining a "history" of each transformation trace, then print them all out:

```hs
import Control.Monad
import Control.Monad.State.Lazy
import Data.List
import Data.Function

test :: StateT [Int] [] ()
test = do
    x'@(x:xs) <- get
    rule <- lift [(+4), (*2), (`rem` 3)]
    put $ (rule x):x'
    return ()

main :: IO ()
main = do
    mapM_ (print . reverse) $ execStateT (replicateM 3 test) [1]
```

```
[1,5,9,13]
[1,5,9,18]
[1,5,9,0]
[1,5,10,14]
[1,5,10,20]
[1,5,10,1]
[1,5,2,6]
[1,5,2,4]
[1,5,2,2]
[1,2,6,10]
[1,2,6,12]
[1,2,6,0]
[1,2,4,8]
[1,2,4,8]
[1,2,4,1]
[1,2,2,6]
[1,2,2,4]
[1,2,2,2]
[1,1,5,9]
[1,1,5,10]
[1,1,5,2]
[1,1,2,6]
[1,1,2,4]
[1,1,2,2]
[1,1,1,5]
[1,1,1,2]
[1,1,1,1]
```

(The initial value is 1; the three transforms are "add 4", "multiply by 2", and "take the remainder mod 3"; we do three transformations in a row. As you would expect, we get `3^3 = 27` results.)

[This page](http://blog.sigfpe.com/2006/10/monads-field-guide.html?m=0) has some nice illustrations of the control flow implied by various stacks of monad transformers. Finding the correct ordering of monad transformers in the stack, and mentally modeling the relevant types, is sometimes nontrivial; nevertheless, they can certainly improve concision in situations that call for them.

## General nondeterminism with generators

Now that the basic idea has been motivated and demonstrated in a language more suited to it (I think this is actually a fairly good illustration of the utility of monads as "programmable semicolons" in languages with good syntax/compiler support for them), we can get to the main question: can we implement a reasonably ergonomic and performant version of this in Python? It is clear enough that McCarthy's `amb` operator can be implemented in basically any programming language; [Rosetta Code](https://rosettacode.org/wiki/Amb#Python) contains several implementations that seem fairly clean and well-behaved, including a (somewhat impractical) transliteration of the list monad described above (as well as more [Haskell](https://rosettacode.org/wiki/Amb#Haskell) examples).

This is somewhat brittle, since we are forced to intermediate every operation in our code TODO. Clearly, we would prefer to go beyond `amb`-like data structures -- we want to interleave Python's control flow with some amount of automated branching logic. Unless we want to either:

- (a) build a DSL supplanting Python's normal constructs and implement our own pseudo-interpreter/compiler; and/or
- (b) perform AST-munging of the kind certain Python JIT compilers (Numba, etc.) do
- (c) perform runtime tracing of the branching structure (more on this later)

...this seemingly requires some way to interrupt execution at specific points and backtrack/rewind to those points, modifying execution state each time before resuming to inject the state of the current "branch".

[Coroutines](https://en.wikipedia.org/wiki/Coroutine) seem well-suited to this purpose; Python's [generators](https://wiki.python.org/moin/Generators), though [not quite the same](https://wiki.c2.com/?GeneratorsAreNotCoroutines), provide enough functionality to do what we have described above. In particular, generators allow us to temporarily stop execution, returning control (and an arbitrary value) to the caller, then later resume execution while passing back ("`send`ing") a new value. We now have the basics of a workable approach: at each "ambiguous" expression in the program, just stop execution, run the *remainder* of the program once with each possible value for that expression, and coalesce the results.

Here is a minimal example of the sort of function we would like to support nondeterminism for; we have a `yield` statement enclosing each "source" of ambiguity/`Amb` expression (and these can use values from prior ones normally, no extra magic needed):

```py
def test(a: int) -> set[((int, int, int), int)]:
    x = yield Amb([1, 2, 3])
    y = yield Amb([2, 4, a])
    if x == y:
        z = yield Amb([0, a * 2])
    else:
        z = 4
    return ((x, y, z), (x + y) * z)
```

We'll implement a very thin class to store these intermediate values and to flag stop points/junctions:

```py
class Amb:
    def __init__(self, xs):
        self.xs = xs
```

Python generators [aren't readily cloneable](https://stackoverflow.com/a/29837018), so we'll write a helper to advance through one `yield` statement for each element of `xs`, using `.send` to set the values we want for each `Amb` expression in order (this is the main source of inefficiency in this implementation):

```py
def send_n(g, xs):
    v = g.send(None)
    try:
        for x in xs:
            v = g.send(x)
        return v
    except StopIteration as v:
        return v.value
```

We catch `StopIteration` to intercept the final return value from the generator. Now we can put together our `amb` decorator, which takes an `Amb`-annotated generator function and returns a function with the nondeterminism effects applied (note that a real-world version of this should probably use `functools.wraps` or similar to copy relevant metadata from the original function: its docstring, name, etc.):

```py
import itertools
from pprint import pprint

def amb(f):
    def go(g, xs):
        if isinstance(v := send_n(g(), xs), Amb):
            return set(itertools.chain.from_iterable(go(g, xs + [x]) for x in v.xs))
        else:
            return set([v])

    def r(*args, **kwargs):
        return go(lambda: f(*args, **kwargs), [])

    return r
```

This does exactly what we described above; `go` merely sends the current "branch" (ordered list of values to pass to the `yield`s/`Amb`s) into the generator, and receives whatever is `yield`ed or `return`ed. If it's a plain value, we are exiting the function and should embed the raw return value in a list. If we get an `Amb`, we are signaling a "breakpoint" or junction, at which we should evaluate *the rest* of the code once for each value inside the `Amb` and concatenate the results (a list of lists).

If we decorate our `test` function with `amb` and evaluate `pprint(list(test(5)))`, we get:

```
[((1, 5, 4), 24),
 ((2, 4, 4), 24),
 ((3, 2, 4), 20),
 ((1, 2, 4), 12),
 ((3, 4, 4), 28),
 ((2, 5, 4), 28),
 ((3, 5, 4), 32),
 ((1, 4, 4), 20),
 ((2, 2, 0), 0),
 ((2, 2, 10), 40)]
```

Excellent!

If you squint, it might be clear that our two branches in `go` map almost directly onto the `bind` (concat) and `return` (singleton) methods of the list monad. Indeed, we could swap in the behavior of a different monad and get the results we expect:

```py
import itertools
from typing import TypeVar, Generic, Tuple
T = TypeVar('T')

class Maybe[T]:
    def __init__(self, x: T, isjust: bool) -> None:
        self.x: T = x
        self.isjust: bool = isjust

    def __str__(self):
        if self.isjust:
            return f"Just({self.x})"
        else:
            return "Nothing"

def send_n(g, xs):
    [elided]

def run(f):
    def go(g, xs):
        if isinstance(v := send_n(g(), xs), Maybe):
            if v.isjust:
                return go(g, xs + [v.x])
            else:
                return Maybe(None, False)
        else:
            return Maybe(v, True)

    def r(*args, **kwargs):
        return go(lambda: f(*args, **kwargs), [])

    return r

@run
def test(a: int) -> Maybe[Tuple[Tuple[int, int, int], int]]:
    x = yield Maybe(7, True)
    y = yield Maybe(3, True)
    if x == a:
        z = yield Maybe(None, False)
    else:
        z = yield Maybe(x + y - 5, True)
    return ((x, y, z), (x + y) * z)


print(5, test(5))
print(7, test(7))
```

```
5 Just(((7, 3, 5), 50))
7 Nothing
```

In the first example, the `x == a` evaluates to `False` and the "monadic state" is set to `Some(x + y - 5)`; in the second, it evaluates to `True` and the state is `Nothing`, which short-circuits evaluation.

(n.b.: this is just illustrative -- another, much better, way to implement something like this if you have a "scalar" short-circuiting monad like `Maybe`/`Option` or `Either`/`Result` is using a custom exception handler that, for example, catches exceptions thrown by `.unwrap` calls on `Nothing`/`Err` values and transforms them back into the appropriate type, basically circumventing the need to actually thread handlers for the wrapper type through your function; this has the benefit of handling deeply nested call stacks with little additional effort)

When we look back at how do-notation desugars in Haskell, the correspondence to the control flow used above is even clearer:

```
do { x1 <- action1
   ; x2 <- action2
   ; mk_action3 x1 x2 }
```

```
action1 >>= (\ x1 -> action2 >>= (\ x2 -> mk_action3 x1 x2 ))
```

(example from https://en.wikibooks.org/wiki/Haskell/do_notation#Translating_the_bind_operator; CC BY-SA 4.0)

Back to the list/set version. This is a nice toy, but is it compatible with more complex control flow? TODO




TODO: compare code with version saved on styx

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

If we are willing to dispense for the moment with some more complex control flow, we may as well just replace the generator-based interceptor with a special object that broadcasts over common arithmetic operations; if we do something like `Wrapped([1, 2]) + Wrapped([3, 4])`, for example, we would expect to get `Wrapped([1 + 3, 2 + 3, 1 + 4, 2 + 4]) == Wrapped([4, 5, 5, 6])`. This is certainly cleaner than with the generator approach; we only traverse the code once, instead of once per branch/combination. The main trouble is with if-statements -- we must follow *both* branches at different places in the array, ideally rewriting the `if` as an `np.where` or similar (we can perform this translation manually, but figuring out how to avoid this is an interesting exercise). We'll come back to that.

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

For/while loops (and conditionals) however only work when the loop condition is "primitive" and does not contain any `Amb`-expressions. For example, a while-loop with a condition derived from an `Amb` would likely not behave as expected; one could imagine a way to make this work by detecting when we drop into a loop, overriding the behavior of `bool`-coercion to keep the loop running until the condition is false for *all* values in the `Amb`, and "masking" assignment operations so that they only affect members of the target value for which corresponding values of any ambiguous expressions in the context still make the loop condition evaluate to `True` (perhaps maintaining a stack of contexts for nested loops/conditional statements), all in an efficient vectorized fashion. This is nontrivial.

In lieu of `guard`, let's add a trivial filtering function (and a restricted variant for concision), and use it to re-implement our Pythagorean triple example from earlier:

TODO

### Basic execution tracing

If we want to support more natural syntax, the problem becomes considerably harder. Let's say we want to support the kind of natural if-statement used in the earlier generator-based examples; as we noted before, we must somehow detect both the expression used as the switching condition and the content of both branches. There is no control flow trick that we can use to change the behavior of the if-statement, so we must substitute it with something more appropriate. To vectorize properly, we have to segment sets (i.e., nondeterministic universes) of input values based on whether they trigger the "true" path or the "false" path and execute the remaining code as though each branch had been taken with only those values.

To achieve this, we can build a simple tracer of the kind used in [JAX](https://github.com/jax-ml/jax) -- this will take a (restricted in terms of the primitives/types/side effects used, and certainly single-threaded, but otherwise arbitrary) function and return a reified computation graph representing its control flow. We said before that we don't want to bother with AST parsing; instead, we'll pass tracer objects that intercept all arithmetic/logical/comparison operations called on them and build up a graph.

Handling conditionals is somewhat harder; the recipe we'll go with is roughly:

- detect any automatic coercion to `bool` (which [happens when conditional statements are evaluated](https://docs.python.org/3/library/stdtypes.html#truth-value-testing)); this expression is used as the switching condition
- once a conditional is detected, spoof the aforementioned expression to `True` to take the corresponding branch; continue tracing the rest of the code
- then, spoof the expression to `False` to take the other branch; continue tracing the rest of the code
- store the remaining trace segments for each of the above in some kind of tree structure
- during execution, we'll evaluate the conditional expression with the actual values of its subexpressions and use this to branch whatever results are produced by the rest of the code

The most obvious issue with this tactic (without further optimization) is that the time complexity of the tracing strategy is `O(2^n)` for `n` conditionals in a function, since we must independently trace everything following the conditional twice (once for the true branch and once for the false branch), as we don't know how they depend on each other structurally; however, we are not actually doing any nontrivial computation during this tracing step, so this is acceptable.

The less obvious issue is that *execution* of the traced function has similar exponential blowup, since the entire universe must be split and evaluated in full each time we encounter a conditional. This is the cost of combining statefulness and arbitrary conditions with vectorization, since it is quite possible for a later conditional to depend on which path *every* prior conditional took; we'll discuss some potential optimizations later. Lastly, it breaks down if coercions happen for other reasons; there are ways to deal with this in practice (alluded to below), but for the purposes of this demonstration we'll just promise to respect the constraints of the implementation.

We will forego loop detection/unrolling, but it is in principle possible using e.g., `__next__`/`__iter__` interception and some of the tools from `sys.settrace`/etc. Since we're demonstrating a tool explicitly targeted at abstracting away iteration/combination logic, this seems like an acceptable sacrifice. One other unfortunate issue is that Python does not allow `and` and `or` to be overridden; one can imagine handling these using a similar strategy to that described above, but for now we'll just use the logical `&` and `|` in their stead.

Let's start with a basic functional tracer that builds up a simple expression tree (but does not handle state or conditionals). To reiterate, we only care about the final return value of the function and are disregarding external side effects for now. We don't want to regard an `Amb` expression used in different places as several different expressions, so we'll use `is` to compare their memory addresses, which should handle aliasing for us. Scalar values will be traced by "infecting" the rest of our code with `__[r]add__`, `__[r]mul__`, etc. (same as with the earlier NumPy examples, but reified into a data structure). Here it is:


TODO


(It is unclear that this actually involved less effort than an AST parser or `settrace`, but perhaps you will find it useful during a CTF or other situation where you don't have those tools available.)

## End

I hope you have learned something useful (or at least entertaining) from this post, or at least found some of the links therein interesting. Thanks for reading!
