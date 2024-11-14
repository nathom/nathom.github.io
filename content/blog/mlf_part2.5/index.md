---
title: "Machine Learning Fundamentals 2.5: Generic behavior in C with macros"
date: 2023-10-02T19:30:41-07:00
draft: false
math: true
comments: true
---

The tradeoff with the understanding gleaned from using a low-level language like C is the lack of features. When writing code for the purpose of demonstration, it's not really an issue. But when you need to write an actual application, it is. <!--more--> One of the notable features missing from C is *compile-time generics*.

Compile time generics are a type of code duplication that depends on the following constraints:

1. There exist multiple classes that implement the same trait
2. There exists code that only depends on those classes through the implemented trait

In this article, I want to show you a way of replicating generics in C, specifically with respect to an operation trait that all the element-wise matrix operations will depend on. We're going to do this with the **only** built-in method of code replication in C: macros.

## What are C macros?

C macros are created with the `#define` directive. They are most simply used like this

```c
#define AN_IMPORTANT_NUMBER 42
```

Which replaces all occurrences of `AN_IMPORTANT_NUMBER` with the literal `42`. You might be wondering: How is it different from this global definition?

```c
const int AN_IMPORTANT_NUMBER = 42;
int main() { ... }
```

Practically, not much. But that's because it's so simple. What if we write a macro function?

```c
#define SQUARED_MACRO(x) x*x
int square_function(int x) { return x*x; }
```

What's the difference between these two? It becomes clear when you think of macros as a slightly more complicated find-and-replace. It is computed by the C preprocessor, and replaces one piece of code with another piece of code at compile time. For example

```c
int x = SQUARED_MACRO(2);
```

expands to the code

```c
int x = 2*2;
```

While the function is not expanded into anything but a *call* instruction. The macro has a couple benefits. First, it doesn't have the overhead of a function call. Second, it is not typed, so `SQAURED_MACRO` can be used with any type that supports multiplication.

The nature of macros that yield these benefits also create a number of dangerous pits of doom. Let's see what `SQUARED_MACRO` expands to in a few interesting cases.

```c
SQUARED_MACRO(2 + 3)
```

```c
2 + 3*2 + 3
```

We want `25`, but got `11`. Not good. We can solve this easily though.

```c
#define SQUARED_MACRO (x)*(x)
```

Another example:

```c
int x = 1, sum = 0;
while (x < 10) { sum += SQUARED_MACRO(x++); }
```

is expanded to

```c
int x = 1, sum = 0;
while (x < 10) { sum += (x++)*(x++); }
```

We see that `x` is incremented twice instead of once! On top of that, it computes `x*(x+1)`, not `x*x`.

So we see that this simple macro spits out complete nonsense for some inputs, which are seen as valid by the compiler. Although the fact that macros *duplicate* code causes these dangerous behaviors, it also lets us be more efficient.

## Goals for the library

I want the library to be minimalist (header only), but complete. We will write functions for a range of operations between two matrices $A$ and $B$.

I classified the necessary functions by the following.

- Allocation type: Should we put the result in a buffer $(C := A \circ B)$, or write it to the first argument $(A = A \circ B)$?
- Loop type: Are we doing a dot product type loop, or an elementwise operation? Should we transpose the second argument $(A \circ B^T)$?
- Elementwise Operation type: Are we Multiplying? Adding? Subtracting?

Notice that all the functions (except dot) are generic over the operation $\circ$.
## Boilerplate

Before we begin, we need to define our matrix type. I'm also going to define a type called `mfloat`, which we can substitute for any float type later.

```c
#define DEBUG 1

typedef double mfloat;

typedef struct {
  mfloat *buf;
  int rows;
  int cols;
} matrix;
```

We're going to assume the elements are stored in `buf` in row-major order. Now, we need some basic getter-setter functions with optional bounds-checking that we'll use in the rest of the library.

```c
static inline mfloat matrix_get(matrix m, int row, int col) {
  if (DEBUG)
    if (!(row >= 0 && col >= 0 && row < m.rows && col < m.cols)) {
      fprintf(
          stderr,
          "matrix_get: Index out of bounds (%d, %d) for matrix "
          "size (%d, %d)\n",
          row, col, m.rows, m.cols);
      exit(1);
    }

  return m.buf[row * m.cols + col];
}

static inline void matrix_set(matrix m, int row, int col,
                              mfloat val) {
  if (DEBUG)
    if (!(row >= 0 && col >= 0 && row < m.rows && col < m.cols)) {
      fprintf(
          stderr,
          "matrix_set: Index out of bounds (%d, %d) for matrix "
          "size (%d, %d)\n",
          row, col, m.rows, m.cols);
      exit(1);
    }

  m.buf[row * m.cols + col] = val;
}
```

And we need a way to print the matrix to the console

```c
static inline void matrix_print(matrix m) {
  for (int i = 0; i < m.rows; i++) {
    printf("[ ");
    for (int j = 0; j < m.cols; j++) {
      // Scientific notation, rounded to 4 decimal places
      printf("%.4e", matrix_get(m, i, j));
      printf(" ");
    }
    printf("]\n");
  }
  printf("\n");
}
```

And a way to allocate and free matrices to, and from, the heap

```c
matrix matrix_new(int rows, int cols) {
  double *buf = calloc(rows * cols, sizeof(double));
  if (buf == NULL) {
    printf("matrix_new: calloc failed.");
    exit(1);
  }

  return (matrix){
      .buf = buf,
      .rows = rows,
      .cols = cols,
  };
}
```

## Dot product

Let's begin with the dot product. I'm only going to write a buffer-allocate version since it is only possible to write the output to the first argument if and only if both of the matrices are square, which we cannot assume.

```c
static inline void matrix_dot(matrix out, const matrix m1,
                              const matrix m2) {
  if (DEBUG)
    if (m1.cols != m2.rows) {
      printf(
          "matrix dot: dimension error (%d, %d) not compat w/ "
          "(%d, %d)\n",
          m1.rows, m1.cols, m2.rows, m2.cols);
      exit(1);
    }
  for (int row = 0; row < m1.rows; row++) {
    for (int col = 0; col < m2.cols; col++) {
      double sum = 0.0;
      for (int k = 0; k < m1.cols; k++) {
        double x1 = matrix_get(m1, row, k);
        double x2 = matrix_get(m2, k, col);
        sum += x1 * x2;
      }
      matrix_set(out, row, col, sum);
    }
  }
}
```

## Element wise operations

Each element-wise operation does the following:

1. Ensure the two matrices have the same dimensions
2. Run operation on each corresponding element
3. Put result in output matrix

We see that the loop will be identical for each of the functions, so let's write a macro for that

```c
#define MAT_ELEMENTWISE_LOOP        \
  for (int i = 0; i < m1.rows; i++) \
    for (int j = 0; j < m1.cols; j++)
```

And a function to check bounds, that just panics if the bounds don't match.

```c
static inline void mat_bounds_check_elementwise(const matrix out,
                                                const matrix m1,
                                                const matrix m2) {
  if (DEBUG)
    if (m1.rows != m2.rows || m1.cols != m2.cols ||
        out.rows != m1.rows || out.cols != m1.cols) {
      fprintf(stderr,
              "Incompatible dimensions for elementwise operation "
              "(%d, %d) & (%d, %d) => (%d, %d) \n",
              m1.rows, m1.cols, m2.rows, m2.cols, out.rows,
              out.cols);
      exit(1);
    }
}
```

Now, we want to implement add, multiply, divide and subtract. Since all the code except the actual computation is identical, we can abstract it into a macro that defines the function.

```c
#define DEF_MAT_ELEMENTWISE_BUF(opname, op)           \
  static inline void matrix_##opname(                 \
      matrix out, const matrix m1, const matrix m2) { \
    mat_bounds_check_elementwise(out, m1, m2);        \
    MAT_ELEMENTWISE_LOOP {                            \
      mfloat x = matrix_get(m1, i, j);                \
      mfloat y = matrix_get(m2, i, j);                \
      matrix_set(out, i, j, op);                      \
    }                                                 \
  }
```

> `##opname` inserts the value of `opname` into the function name.

Looking ahead, we know we will have to define add, multiply, divide, subtract functions for all the variations, so let's write a macro that does that for a given function-defining-macro

```c
#define DEF_ALL_OPS(OP_MACRO) \
  OP_MACRO(sub, (x - y));     \
  OP_MACRO(add, (x + y));     \
  OP_MACRO(div, (x / y));     \
  OP_MACRO(mul, (x * y));
```

Now we can actually define the 4 functions!

```c
DEF_ALL_OPS(DEF_MAT_ELEMENTWISE_BUF)
```

Boom! In one line, we have defined the functions `matrix_add`, `matrix_sub`, `matrix_div`, and `matrix_mul`. 

Now let's try implementing in-place operations.

```c
static inline void mat_bounds_check_elementwise_ip(
    matrix m1, const matrix m2) {
  if (DEBUG)
    if (m1.rows != m2.rows || m1.cols != m2.cols) {
      fprintf(stderr,
              "Incompatible dimensions for elementwise in-place "
              "operation (%d, %d) & (%d, %d) \n",
              m1.rows, m1.cols, m2.rows, m2.cols);
      exit(1);
    }
}

#define DEF_MAT_ELEMENTWISE_IP(opname, op)                 \
  static inline void matrix_ip_##opname(matrix m1,         \
                                        const matrix m2) { \
    mat_bounds_check_elementwise_ip(m1, m2);               \
    MAT_ELEMENTWISE_LOOP {                                 \
      mfloat x = matrix_get(m1, i, j);                     \
      mfloat y = matrix_get(m2, i, j);                     \
      matrix_set(m1, i, j, op);                            \
    }                                                      \
  }
```

With this new macro, we can just do

```c
DEF_ALL_OPS(DEF_MAT_ELEMENTWISE_IP)
```

and `matrix_ip_add`, `matrix_ip_mul`, etc. are defined. We can also do this for transpose operations, but I won't show it here.

## Unary operations

Sometimes we want to do a unary operation on a matrix $A$, such as scalar $A := A^2$ or $A := -A$. Let's make a unary function generic over the operation.

```c
#define DEF_MAT_UNARY_IP(opname, op)            \
  static inline void matrix_ip_##opname(matrix m1) { \
    MAT_ELEMENTWISE_LOOP {                           \
      mfloat x = matrix_get(m1, i, j);               \
      matrix_set(m1, i, j, op);                      \
    }                                                \
  }

DEF_MAT_UNARY_IP(square, (x * x))
DEF_MAT_UNARY_IP(negate, (-x))
DEF_MAT_UNARY_IP(sqrt, (sqrt(x)))
```

And now `matrix_ip_square(A)`, `matrix_ip_negate(A)`, etc. are defined.


## Conclusion

Just like that, we "wrote" an entire matrix operation library with minimal effort. But this begs the question: Is this code safe? 

As long as you `#undef` all the macros that you defined, yes, it is just as safe as writing all the functions yourself. On the other hand, if you expose macros as part of your library's functionality, it may no longer be safe. But just because some code is safe **does not** mean it is something you should do. If there is an issue with the code, it may be difficult to debug since you can't actually see what it is expanded to. So in a production environment, the use of macros is strongly discouraged. I just used it here because it was an easy-way-out for an educational series.

If you have any questions or suggestions, feel free to leave a comment or shoot me an email. Thanks for reading.
