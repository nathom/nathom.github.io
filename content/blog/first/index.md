---
title: "The Basel Problem"
date: 2023-07-28T22:21:54-07:00
draft: false
math: true
---

Here are some code snippets in various languages that compute the [Basel Problem](https://en.wikipedia.org/wiki/Basel_problem):



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

Let's see how they compare in performance on an M1 Pro, for large N.

$N=10^8$

| Language           | Time (s)   |
| ------------------ | ---------- |
| Rust (--release)   | 0.095      |
| C  (-O3)             | 0.100      |
| Haskell (-O3)      | 1.35       |
| Python (3.10)      | 7.15       |
