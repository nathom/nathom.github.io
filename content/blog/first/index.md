---
title: "The Basel Problem"
date: 2023-07-28T22:21:54-07:00
draft: false
math: true
katex: true
---

Here are some code snippets in various languages that compute the [Basel Problem](https://en.wikipedia.org/wiki/Basel_problem):

<!--more-->

To begin, $\LaTeX$

$$
\sum_{n=1}^\infty \frac{1}{n^2} = \frac{\pi^2}{6} = 1.6449340668482264
$$

## Python

```python
def pi_squared_over_6(N: int) -> float:
    return sum(x**(-2) for x in range(1,N))
```

## Rust

```rust
fn pi_squared_over_6(N: u64) -> f64 {
    (1..N).map(|x| 1.0 / ((x*x) as f64)).sum()
}
```

## Haskell

```haskell
piSquaredOver6 :: Integer -> Double
-- no capital N in Haskell :(
piSquaredOver6 n = sum $ map (\x -> 1 / fromIntegral (x * x)) [1..n]
```

## C

```c
double pi_squared_over_6(unsigned int N) {
    double sum = 0.0;
    for (int i = 1; i < N; i++) {
        sum += 1.0 / (i*i);
    }
    return sum;
}
```

What's your favorite solution?

## Performance

Let's see how they compare in performance on an M1 Pro, for $N=10^9$.

| Language            | Time (ms, $\mu \pm \sigma$) |
| ------------------  | ----------                  |
| Rust (parallelized) | $112.6 \pm 3.5$             |
| Rust (--release)    | $937.9 \pm 0.4$             |
| C  (-O3)            | $995.3 \pm 0.8$             |
| Haskell (-O3)       | $13454 \pm 205$             |
| Python (3.10)       | $67720 \pm 0$                        |

## Fixing Python

The python code took an absurdly long amount of time to run, so lets fix it
by taking advantage of numpy, which calls into vectorized C code.

```python
import numpy as np

def pi_squared_over_6(N: int) -> float:
    x = np.ones(N)
    r = np.arange(1,N)
    sq = np.square(r)
    div = np.divide(x, sq)
    return float(np.sum(div))
```

A bit better, but when I'm checking `btm`, the excessive memory consumption
suggests most of the work done is moving around billions of floats,
not the actual arithmetic. Lets try splitting this into chunks:


```python
def pi_squared_over_6(N: int) -> float:
    CHUNKS = 25000
    SIZE = N // CHUNKS
    s = 0.0
    x = np.ones(N // CHUNKS - 1)
    for i in range(CHUNKS):
        N_tmp = i * SIZE
        r = np.arange(N_tmp + 1, N_tmp + SIZE)
        sq = np.square(r)
        div = np.divide(x, sq)
        s += np.sum(div)
        # deallocate memory
        del sq
        del div
        del r
        
    return s
```

Much better! Now it's running under 2 seconds!
