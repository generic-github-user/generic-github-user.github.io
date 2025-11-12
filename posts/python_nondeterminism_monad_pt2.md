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

The `Range` helper highlights an issue with this approach to vectorization: TODO


(It is unclear that this actually involved less effort than an AST parser or `settrace`, but perhaps you will find it useful during a CTF or other situation where you don't have those tools available.)
