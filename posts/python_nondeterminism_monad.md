# Nondeterminism monad in Python

*This post contains some fairly lengthy exposition; if you want to skip right to the content promised by the title, go to `## Nondeterminism in Python`*

Many of the most curious and useful features of functional programming languages like Haskell derive from their ability (often unencumbered by the norms and constraints of industrial software engineering) to restate common algorithmic problems in novel ways -- to perform a change of basis into a domain more suited to the problem. One such frame shift (or rather, category of such) is widely known as [*declarative programming*](https://en.wikipedia.org/wiki/Declarative_programming) (as opposed to imperative or functional programming, for example), and concerns programming languages, libraries, and techniques based on stating the problem domain or constraint system, as well as the desired objective or target (the "what"), at a high level and leaving the low-level algorithmic details to the optimizer or runtime (the "how"). In some cases this may take the form of a domain-specific optimization or constraint solving library; other times it is integrated more tightly with a language's semantics and execution model.

One self-contained and useful tool from this paradigm is "nondeterminism" (this is a somewhat overloaded term, but in this case I am *not* talking about the kind of nondeterminism that people mention with respect to e.g., reproducibility of software artifacts or experiments). The premise is that we delineate the ways in which a program can branch or the alternatives to be selected between (potentially with nesting, recursion, and other complications) and search/"solve" for some solution among the set of possible paths or choices. That is to say, the nondeterminism interface should abstract away the construction of the search space and execution of the search procedure to some extent; the programmer need only be concerned with which choices are available and how they interact (e.g., how state evolves over time depending on the branch taken at each step).

The classical implementation of this scheme is described in chapter 4.3 of the well-known *Structure and Interpretation of Computer Programs*: McCarthy's `amb` ("ambiguous") operator, usually implemented in Lisp. I will omit the details of the implementation since there are [decent](https://rosettacode.org/wiki/Amb) [explanations](https://www.randomhacks.net/2005/10/11/amb-operator/) [elsewhere](https://ds26gte.github.io/tyscheme/index-Z-H-16.html), but the idea rhymes with what I've described above: we consider "splitting" execution into different paths corresponding to available combinations of initial values, and return results only from the path (or paths) where execution "succeeded" (some specified criteria was met).

[Haskell](https://www.haskell.org/) has what I would consider a somewhat more principled implementation of nondeterminism using monads. In particular, the built-in list type forms a monad, with `\xs, f -> concat (map f xs)` as the `>>=` (`bind`) operation (and the singleton list constructor, i.e., `return x = [x]`, as `return`/`pure`). This means that the resulting list will be constructed by passing each element in `xs` to `f` to yield a new list, then concatenating the results:

```hs
ghci> [1, 2, 3] >>= (\x -> [x * 2, x *3])
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

It is important to note that -- unlike with naive implementations that iterate over all combinations of elements from a handful of statically known sets, and only check which combinations would have survived at the end -- this approach really does prune branches each time `guard` is invoked, avoiding much unnecessary work, and has all the execution semantics you would expect of a handwritten "iterate over items in source collections -> map transformations over each element and collect the results -> filter/prune -> ..." approach.

(The main deficiency, aside from those mild to moderate performance concerns (cache locality, laziness) that apply to Haskell's execution model more generally, is that since `Data.Set` cannot be made into a monad (for any `Set a`, `a` carries an `Ord` constraint), we cannot use the obvious optimization strategy: "implement nondeterminism backed using a `Set` so that at each step, the universe is automatically collapsed down into unique values". There are some packages which claim to implement a performant, monad-compatible set datatype, the implementation details of which I know not.)

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

Now that the basic idea has been motivated and demonstrated in a language more suited to it, we can get to the main question: can we implement a reasonably ergonomic and performant version of this in Python? It is clear enough that McCarthy's `amb` operator can be implemented in basically any programming language; [Rosetta Code](https://rosettacode.org/wiki/Amb#Python) contains several implementations that seem fairly clean and well-behaved, including a (somewhat impractical) transliteration of the list monad described above (as well as more [Haskell](https://rosettacode.org/wiki/Amb#Haskell) examples).

Clearly, we would prefer to go beyond `amb`-like data structures -- we want to interleave Python's control flow with some amount of automated branching logic. Unless we want to either

- (a) build a DSL supplanting Python's normal constructs and implement our own pseudo-interpreter/compiler; and/or
- (b) perform AST-munging of the kind Python JIT compilers (JAX, Numba, etc.) have to do

...this seemingly requires some way to interrupt execution at specific points and backtrack/rewind to those points, modifying execution state each time before resuming to inject the state of the current "branch".

[Coroutines](https://en.wikipedia.org/wiki/Coroutine) seem well-suited to this purpose; Python's [generators](https://wiki.python.org/moin/Generators), though [not quite the same](https://wiki.c2.com/?GeneratorsAreNotCoroutines), provide enough functionality to do what we have described above. In particular, generators allow us to temporarily stop execution, returning control (and an arbitrary value) to the caller, then later resume execution while passing back a new value. We now have the basics of a workable approach: at each "ambiguous" expression in the program, just stop execution, run the *remainder* of the program once with each possible value for that expression, and coalesce the results.

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

We catch `StopIteration` to intercept the final return value from the generator. Now we can put together our `amb` decorator, which takes an `Amb`-annotated generator function and returns a function with the nondeterminism effects applied (note that a real-world version of this should probably use `functools.wraps` or similar to copy relevant metadata from the original function; its docstring, name, etc.):

```py
import itertools

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

This does exactly what we described above; `go` merely sends the current "branch" (ordered list of values to pass to the `yield`s/`Amb`s) . If you squint, it might be clear that our two branches in `go` map directly onto the `bind` (concat) and `return` (singleton) methods of the list monad. Indeed, we could swap in the behavior of a different monad and get the results we expect:

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
    v = g.send(None)
    try:
        for x in xs:
            v = g.send(x)
        return v
    except StopIteration as v:
        return v.value

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

When we look back at how do-notation desugars in Haskell, the correspondence is even more clear:

```
do { x1 <- action1
   ; x2 <- action2
   ; mk_action3 x1 x2 }
```

```
action1 >>= (\ x1 -> action2 >>= (\ x2 -> mk_action3 x1 x2 ))
```

(example from https://en.wikibooks.org/wiki/Haskell/do_notation#Translating_the_bind_operator; CC BY-SA 4.0)

Back to the list/set version.
