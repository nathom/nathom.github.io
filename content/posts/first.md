---
title: "Testing..."
date: 2023-07-28T22:21:54-07:00
draft: true
---


# Welcome to my blog

I plan on posting random thoughts here. Maybe some tutorials for subjects I find interesting.

Here is some diverse fontmatter:

```python
def pi_squared_over_6(N: int) -> float:
    return sum(x**(-2) for x in range(1,N))
```

```rust
fn pi_squared_over_6(N: u64) -> f64 {
    (1..N).map(|x| 1.0 / ((x * x) as f64)).sum()
}
```

```haskell
piSquaredOver6 :: Integer -> Double
piSquaredOver6 N = sum $ map (\x -> 1 / fromIntegral (x * x)) [1..N]
```

```c
double pi_squared_over_6(unsigned int N) {
    double sum = 0.0;
    for (int i = 1; i < N; i++) {
        sum += 1.0 / (i*i);
    }
    return sum;
}
```

$$
\sum_{n=1}^\infty \frac{1}{n^2} = \frac{\pi^2}{6} \approx 1.6449340668482264
$$

Fancy math:

$$
f(a) = \oint_\gamma \frac{f(z)}{z-a} dz
$$


I am $\frac{1}{3}$ of the way done with this site.
