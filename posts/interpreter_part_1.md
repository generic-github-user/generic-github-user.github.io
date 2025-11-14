---
title: Building an interpreter for an object-oriented programming language (part 1)
tags: [haskell, parsers, PL, interpreters]
location: Maryland
---

This blog series will walk step-by-step through the implementation of a basic interpreter for a simple object-oriented language in which (nearly) all values behave like "objects" in a typical weakly & dynamically typed programming language. It aims to (among other things) introduce the reader to (or provide further insight into) parser combinators in Haskell, the semantics of "JavaScript-like" programming languages, basic compiler optimizations, performance considerations in functional programming languages and language runtimes, and other topics relating to these. Some disclaimers before we start:

- I am not a professional and do not have very deep knowledge of the subjects herein; I will do my best to communicate a level of epistemic confidence consistent with my actual experience, but this is just as much a pedagogical endeavor for me as for the hypothetical reader
- The methods and code samples shown in these blogs are not intended to be in any way "production-ready", but rather merely illustrative -- most of them will have one or many serious shortcomings, which I'll try to explicitly address when relevant
- I will avoid lingering on definitions, as there are already many excellent definitions out there for most of the terms I'll use (I'll link to those when I can); if you are still unclear on what a term means, Google it, ask your preferred LLM, or [email me](/contact)

With this out of the way, we can discuss what we plan to do and why someone might want to do something similar. We're aiming to construct a reasonably performant interpreter for a (relatively) simple programming language with syntax and semantics most closely resembling JavaScript, but not far removed from a language like Lua or Python. These languages are notable mainly for being "dynamic" and having relatively weak type systems with little to no upfront (i.e., "compile-time") enforcement of type system invariants; and are relevant to our project also because their canonical implementations are interpreters rather than compilers. All three are ubiquitous in different niches: JavaScript in websites and web applications (including e.g., Electron applications); Python in a wide range of "backend" systems and machine learning settings; and Lua as an embedded scripting language in games and other software.

Though I am generally a proponent of strong, static type systems for enforcing invariants at compile time and ["making invalid states unrepresentable"](https://lexi-lambda.github.io/blog/2019/11/05/parse-don-t-validate/), these kinds of languages are more flexible in many ways, and useful for scripting or prototyping. They are also useful pedagogically, and more suitable for this blog post series since they allow us to circumvent e.g., having to lay out a formalization of an elaborate type system. As for constructing an interpreter rather than a compiler: the former is often considered much simpler

Here is a rough sketch of the language I have in mind:

- 
- all values, including functions, lists/arrays, and "primitives" (integers, floats, booleans, etc.) should have an "object-like" interface exposed as if they were simple dictionaries, even if the underlying representation in the interpreter doesn't treat them the same way as user-constructd compound objects
- blocks in functions, conditionals, and match statements will have "Rust-like" semantics, in which most or all statements are also expressions (evaluate to a concrete value), and if the last statement (or expression) in a block returns a value, the entire block evaluates to this value
- a syntax for anonymous ("lambda") functions is available
- functions (including anonymous ones) defined inside other functions can "capture" values in their environment when they are created, creating [closures](https://en.wikipedia.org/wiki/Closure_(computer_programming)); functions should be first-class and able to be passed around as values, received by/returned from functions, assigned to variables/object fields, converted to strings, etc.
- there will be no class hierarchy or trait/interface/typeclass system built into the language, though the facilities necessary to implement it as a library should be available
- the evaluation semantics should be left up to the language implementation, within reason -- in particular, the interpreter (or compiler) should decide when to perform lazy or strict evaluation, in general taking whatever choice(s) will maximize performance
- no manual memory management should be required of the programmer; where necessary, allocations and deallocations should be handled by the language runtime (probably via [garbage collection](https://en.wikipedia.org/wiki/Garbage_collection_(computer_science)))
