# Coding Rules

## Principles

1. **Comments explain WHY, not WHAT.** Code itself documents WHAT. Comments document reasoning, context, non-obvious constraints — why this approach over alternatives.

2. **Functions are self-explanatory and have one responsibility.** Each function does one thing, named to describe that thing. If it needs "and" in the name, split it.

3. **Test behavior, not implementation.** Test the public contract. Don't mock internals or assert on private state. Refactoring shouldn't break green tests.

4. **Interfaces are deep.** An interface should minimize the learning cost (concept surface area) while maximizing the power it delivers. Prefer few, powerful operations over many small ones. Simpler to use, harder to misuse.

   Deep example:
   ```c
   Result process(int a, float b);
   ```
   One call. Hides init/set/compute/cleanup. Caller pays little mental cost per unit of functionality.

   Shallow example:
   ```c
   void init(Foo*);
   void set_a(Foo*, int);
   void set_b(Foo*, float);
   void compute(Foo*);
   ```
   4 functions, each trivial. Caller must know internal fields, call order, lifecycle. Lots of concept surface, little power per call.
