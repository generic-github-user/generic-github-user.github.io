---
title: Nondeterminism monad in Python
tags: [haskell, monads, python, generators, tricks]
location: Maryland and NYC
---

*This post contains some fairly lengthy exposition; if you want to skip right to the content promised by the title, go to [`## General nondeterminism with generators`](#general-nondeterminism-with-generators)*

Many of the most curious and useful features of functional programming languages like Haskell derive from their ability (often unencumbered by the norms and constraints of industrial software engineering) to restate common algorithmic problems in novel ways -- i.e., to perform a change of basis into a domain more suited to the problem. One such frame shift (or rather, category of such) is widely known as [*declarative programming*](https://en.wikipedia.org/wiki/Declarative_programming) (as opposed to imperative or functional programming, for example), and concerns programming languages, libraries, and techniques based on stating the problem domain or constraint system, as well as the desired objective or target (the "what"), at a high level and leaving the low-level algorithmic details to the optimizer or runtime (the "how"). In some cases this may take the form of a domain-specific optimization or constraint solving library; other times it is integrated more tightly with a language's semantics and execution model.

One self-contained and useful tool from this paradigm is ["nondeterminism"](https://en.wikipedia.org/wiki/Nondeterministic_programming) (this is a somewhat overloaded term, but in this case I am *not* talking about the kind of nondeterminism that people mention with respect to e.g., reproducibility of software artifacts or experiments). The premise is that we delineate the ways in which a program can branch or the alternatives to be selected between (potentially with nesting, recursion, and other complications) and search/"solve" for some solution among the set of possible paths or choices. That is to say, the nondeterminism interface should abstract away the construction of the search space and execution of the search procedure to some extent; the programmer need only be concerned with which choices are available and how they interact (e.g., how state evolves over time depending on the branch taken at each step).

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
...
```

(There is a tremendous amount of fascinating monad/applicative/traversable/alternative machinery that works with almost all of Haskell's basic types, and which I would recommend having a look at if the above interests you at all; another example I'm fond of is `sequence [[1..2], [3..6]]`, which exploits the fact that lists are both `Traversable` and `Monad`.)

It is important to note that -- unlike with naive implementations that iterate over all combinations of elements from a handful of statically known sets, and only check which combinations would have survived at the end -- this approach really does prune branches each time `guard` is invoked, avoiding much unnecessary work, and has all the execution semantics you would expect of a handwritten "iterate over items in source collections -> map transformations over each element and collect the results -> filter/prune -> ..." approach.

(The main deficiency, aside from those mild to moderate performance concerns (cache locality, laziness) that apply to Haskell's execution model more generally, is that since `Data.Set` cannot be made into a monad (for any `Set a`, `a` carries an `Ord` constraint), we cannot use the obvious optimization strategy: "implement nondeterminism backed using a `Set` so that at each step/branching point, the universe is automatically collapsed down into unique values". There are [some packages](https://hackage.haskell.org/package/set-monad) which claim to implement a performant, monad-compatible set datatype, the implementation details of which I know not.)

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

[This page](http://blog.sigfpe.com/2006/10/monads-field-guide.html?m=0) has some nice ileustrations of the control flow implied by various stacks of monad transformers. Finding the correct ordering of monad transformers in the stack, and mentally modeling the relevant types, is sometimes nontrivial; nevertheless, they can certainly improve concision in situations that call for them.

## General nondeterminism with generators

Now that the basic idea has been motivated and demonstrated in a language more suited to it (I think this is actually a fairly good illustration of the utility of monads as "programmable semicolons" in languages with good syntax/compiler support for them), we can get to the main question: can we implement a reasonably ergonomic and performant version of this in Python? It is clear enough that McCarthy's `amb` operator can be implemented in basically any programming language; [Rosetta Code](https://rosettacode.org/wiki/Amb#Python) contains several Python implementations that seem fairly clean and well-behaved, including a (somewhat impractical) transliteration of the list monad described above (as well as more [Haskell](https://rosettacode.org/wiki/Amb#Haskell) examples).

This is somewhat brittle, since we are generally forced to intermediate every operation in our code with a set of predefined combinators. Clearly, we would prefer to go beyond `amb`-like data structures -- we want to interleave Python's control flow with some amount of automated branching logic. Unless we want to either:

- build a DSL supplanting Python's normal constructs and implement our own pseudo-interpreter/compiler; and/or
- perform AST-munging of the kind certain Python JIT compilers (Numba, etc.) do
- perform runtime tracing of the branching structure (more on this in an in-progress future post, potentially)

...this seemingly requires some way to interrupt execution at specific points and backtrack/rewind to those points, modifying execution state each time before resuming to inject the state of the current "branch".

[Coroutines](https://en.wikipedia.org/wiki/Coroutine) seem well-suited to this purpose; Python's [generators](https://wiki.python.org/moin/Generators), though [not quite the same](https://wiki.c2.com/?GeneratorsAreNotCoroutines), provide enough functionality to do what we have described above. In particular, generators allow us to temporarily stop execution, returning control (and an arbitrary value) to the caller, then later resume execution while passing back ("`send`ing") a new value. We now have the basics of a workable approach: at each "ambiguous" expression in the program, just stop execution, run the *remainder* of the program once with each possible value for that expression, and coalesce the results. The actual code implementing this turns out to be quite short.

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

Let's try recreating our earlier `StateT` example as faithfully as possible:

```py
@amb
def statet_test(start: int) -> list[list[int]]:
    r = [start]
    for i in range(3):
        rule = yield Amb([lambda x: x + 4, lambda x: x * 2, lambda x: x % 3])
        start = rule(start)
        r.append(start)
    return r

for r in statet_test(1):
    print(r)
```

Surprisingly, this is ~as concise as the Haskell version (albeit much slower)! I've temporarily changed the backing type in `go` from a `set` to a `list`, for two reasons: we want to get our results in the same order and we cannot have a `set[list[int]]` since lists are non-hashable. Either one works fine. Here are the results (seemingly matching the ones from the original):

```
[1, 5, 9, 13]
[1, 5, 9, 18]
[1, 5, 9, 0]
[1, 5, 10, 14]
[1, 5, 10, 20]
[1, 5, 10, 1]
[1, 5, 2, 6]
[1, 5, 2, 4]
[1, 5, 2, 2]
[1, 2, 6, 10]
[1, 2, 6, 12]
[1, 2, 6, 0]
[1, 2, 4, 8]
[1, 2, 4, 8]
[1, 2, 4, 1]
[1, 2, 2, 6]
[1, 2, 2, 4]
[1, 2, 2, 2]
[1, 1, 5, 9]
[1, 1, 5, 10]
[1, 1, 5, 2]
[1, 1, 2, 6]
[1, 1, 2, 4]
[1, 1, 2, 2]
[1, 1, 1, 5]
[1, 1, 1, 2]
[1, 1, 1, 1]
```

Just to round it out, here's a more interesting example using string manipulation:

```py
from random import shuffle

@amb
def string_test() -> list[str]:
    a = yield Amb(["where [content] go",
                   "what [content] do",
                   f"{yield Amb(['how', 'why'])} [content] do it"])
    content = (yield Amb(['will', 'did'])) + ' ' + (yield Amb(['you', 'she', 'he']))
    return a.replace('[content]', content) + (yield Amb(["?", "...?"]))

s = string_test()
shuffle(s)
pprint(s[:20])
```

```
['where did he go?',
 'what will he do...?',
 'why did you do it?',
 'what did you do...?',
 'why will he do it...?',
 'what will you do...?',
 'where will you go...?',
 'how did he do it?',
 'why did he do it?',
 'where will you go...?',
 'where will she go...?',
 'where did she go?',
 'what did you do?',
 'why will you do it?',
 'why did he do it...?',
 'where did she go...?',
 'how will she do it?',
 'how did she do it?',
 'what did she do?',
 'what will he do?']
```

If you squint at the definitions, it might be clear that our two branches in `go` map almost directly onto the `bind` (concat) and `return` (singleton) methods of the list monad. Indeed, we could swap in the behavior of a different monad and get the results we expect:

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

(n.b.: this is just illustrative -- another, much better, way to implement something like this if you have a "scalar" short-circuiting monad like `Maybe`/`Option` or `Either`/`Result` is by using a custom exception handler that, for example, catches exceptions thrown by `.unwrap` calls on `Nothing`/`Err` values and transforms them back into the appropriate type, basically circumventing the need to actually thread handlers for the wrapper type through your function; this has the benefit of handling deeply nested call stacks with little additional effort, and is almost surely faster)

When we look back at how do-notation desugars in Haskell, the correspondence to the control flow used above is even clearer:

```hs
do { x1 <- action1
   ; x2 <- action2
   ; mk_action3 x1 x2 }
```

```hs
action1 >>= (\ x1 -> action2 >>= (\ x2 -> mk_action3 x1 x2 ))
```

(example from [Wikibooks: "Haskell/do notation"](https://en.wikibooks.org/wiki/Haskell/do_notation#Translating_the_bind_operator); CC BY-SA 4.0)

It's also quite possible to define our own `guard` in analogy to Haskell's (here we again use a list instead of a set to preserve iteration order, for clarity); again, branches for which the test condition fails will not be executed:

```py
guard = lambda c: Amb([None]) if c else Amb([])

@amb
def guard_test(a: int) -> list[int]:
    x = yield Amb(range(0, a))
    y = yield Amb(range(a, a * 2))
    yield guard((x + y) % 2 == 0)
    return (x, y, x + y)

pprint(list(guard_test(10)))
```

```
[(0, 10, 10),
 (0, 12, 12),
 (0, 14, 14),
 (0, 16, 16),
 (0, 18, 18),
 (1, 11, 12),
 (1, 13, 14),
 (1, 15, 16),
 (1, 17, 18),
 (1, 19, 20),
 (2, 10, 12),
 (2, 12, 14),
 ...
```

As a final note, you can probably convince yourself without too much effort that the implicit control flow in cases where we select specific branches (i.e., perform goal-directed search) directly mirrors the backtracking that would occur in say, an equivalent hand-programmed tree search algorithm, or a typical [continuation-based](https://ds26gte.github.io/tyscheme/index-Z-H-16.html) implementation of `amb` in Lisp. Wikipedia's [description](https://en.wikipedia.org/wiki/Nondeterministic_programming) of this process is somewhat more precise:

> If all alternatives fail at a particular choice point, then an entire branch fails, and the program will backtrack further, to an older choice point. One complication is that, because any choice is tentative and may be remade, the system must be able to restore old program states by undoing side-effects caused by partially executing a branch that eventually failed.

Back to the list/set version. This is a nice toy, but is it compatible with more complex control flow, say, recursion? It's not hard to imagine that it might be -- there's no external state attached to the function itself, and generators in Python are distinct state-management objects created by generator functions, not the functions themselves. Here is one way (a trivial example, but not hard to extend):

```py
@amb
def recursion_test(a: int) -> list[int]:
    if a < 5:
        return (yield Amb([a]))
    x = yield Amb(range(a - 4, a))
    return (a, recursion_test(x))

pprint(recursion_test(7))
```

```
[(7, [3]),
 (7, [4]),
 (7, [(5, [1]), (5, [2]), (5, [3]), (5, [4])]),
 (7,
  [(6, [2]),
   (6, [3]),
   (6, [4]),
   (6, [(5, [1]), (5, [2]), (5, [3]), (5, [4])])])]
```

(We could also, if we wished, modify our decorator to "flatten" the results from recursive calls, so that the semantics more closely resemble "explore all possible combinations of inputs in all execution frames, coalescing the outcomes into a single collection.")

The main issue here (and one which is hopefully evident by now) is that this is not very performant, both due to Python being an exceptionally slow language and the overhead of needing to reproduce the generator state de novo for each path we evaluate. One possible approach that solves both issues is representing ambiguous/nondeterministic variables, intermediate expressions, and outputs using suitably shaped NumPy arrays, and broadcasting computations to recreate our "compute this operation for every pair of inputs generated by these two ambiguous expressions" semantics. This runs into some issues with choosing how to model things like a range of values where one or both bounds is an `Amb` value, representing conditionals and loops without inflicting onerous DSL syntax on the programmer or depriving them of the other facilities of the language, performance overhead from fitting branching/nonuniform computations into neat rectangular ndarrays, and representing non-primitive datatypes; it is nevertheless not a bad option for certain kinds of tasks. (Another is using Haskell with `-O3` set in GHC.)

I plan to address some of this in more detail in a future post, but for now I've opted to end it here in the interest of being able to get this out in a timely manner. Hopefully I've provided enough context to get you thinking about how some of the aforementioned issues might be addressed.

I hope you have learned something useful (or at least entertaining) from this post, or at least found some of the links therein interesting. Thanks for reading!
