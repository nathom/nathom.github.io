---
title: "Machine Learning Fundamentals Part 2: Multivariate Linear Regression"
date: 2023-09-20T10:44:52-07:00
draft: false
toc: true
comments: true
math: true
socialShare: true
---


Welcome to Part 2 / N of my Machine Learning Fundamentals in C series! If you haven't already, go through [part 1](/blog/mlf_part1)—I'm going to assume you're familiar with the concepts there in this article.

<!--more-->

## Beyond one variable

Our first machine learning model, the Univariate Linear Regression Model, was *technically* machine learning, but if we're being honest with ourselves, it's not very impressive. Then what was the point of going through all that trouble? To set the foundation for more complicated models, like the *multivariable* linear model, which we'll be looking at today.

## The Multivariable Linear Model

The Multivariate Linear Model (MLM) is essentially the same as its Univariate analogue, except it allows us to have multiple input variables that influence the output. In the last part, we tried to predict housing prices based solely on square footage. We all know that a 5000 $\text{ft}^2$ house in Arkansas is many times cheaper than one in San Jose, but our model would spit out the same price for both.

This is where we realize that we need our model to be able to mix-and-mash variables of different classes, each of which may have a different effect on our prediction. 

## Problem Statement

Let's first define our problem: 

> We are given data for the following *input variables* and their corresponding house prices $y_n$ (in $1000s):
>
> $x_1$: Square footage (in $\text{ft}^2$)
>
> $x_2$: Lot width (in ft)
>
> $x_3$: Lot depth (in ft)
>
> $x_4$: Distance from shore (in miles)
>
> Predict the price of a house given a new set of data.


We know that each of these factors will have an effect on the price. Some (like square footage and distance from shore) will have a very large effect, while others will have a smaller one.

So how can we express these relationships as a model? Let's try to construct one by looking at some plots, and guesstimating.

### Plot: Square footage vs. Price 

{{< plotly src="sqft_plot.html" >}}


We can see that there is a positive association between square footage and price. The trend line is modeled by $0.34 x_1 + 550$.

### Plot: Lot width vs. Price


{{< plotly src="width_plot.html" >}}

There is also a slightly positive association here. The trend line is $3.3 x_2 + 1771$.

### Plot: Lot depth vs Price

{{< plotly src="depth_plot.html" >}}

This has a much stronger positive association than lot width, suggesting a deeper lot shape is valued higher by homebuyers. The trend line is $5.84 x_3 + 1459$.

### Plot: Distance from shore vs. Price

{{< plotly src="shore_plot.html" >}}

As we would expect, the further away from the shore, the lower the price. We also see that the effect diminishes as we go further from the shore. The trend line is $-8.07 x_4 + 2758$.

### A First Guess for the Model

The trendlines tell us the approximate relationship between each feature and the housing price. We can mash these together to get an approximate model of the data. Let's try adding all of the trendlines together and see if we get a reasonable model.

$$
\begin{aligned}
\hat{y} &= 0.34 x_1 + 550 + 3.3 x_2 + 1771 + 5.84x_3 + 1459 - 8.07 x_4 + 2758 \\\
 &= 0.34 x_1 + 3.3 x_2  + 5.84x_3  - 8.07 x\_4  + 6538
\end{aligned}
$$

Intuitively, what does this mean? We have a base value (bias) of $6.5M, which represents the theoretical value of a house with 0 sqft, no lot, at the oceanside. Wait... that does't sound right. We should probably take the average of the y-intercepts because each plot overlaps with the others. New model:


$$
\begin{align*}
\hat{y} &= 0.34 x_1  + 3.3 x_2  + 5.84x_3  - 8.07 x_4  + (550 + 1771 + 1459 + 2758)/4 \\\\
 &= 0.34 x_1 + 3.3 x_2  + 5.84x_3  - 8.07 x_4  + 1634.5 \\\\
\end{align*}
$$

Ok... Now its $1.6M. But maybe this is reasonable since it is right at the shore. The rest of the model seems to make sense. It suggests that

| For every 1 unit increase in | housing price changes by |
| ---------------------------- | ------------------------ |
| Square footage               | $340                     |
| lot width                    | $3300                     |
| lot depth                  | $5840                     |
| distance from shore          | -$8070                  |

Though reasonable, these are *just assumptions* based on some shoddy math. 
To truly solve our problem, we need to find the *optimal* MLM that predicts housing prices given these 4 features. In math terms, we need to find the weights $w_1, w_2, w_3, w_4$  and bias $b$ such that

$$
\hat{y} = w_1 x_1 + w_2x_2 + w_3 x_3 + w_4 x_4 + b
$$

predicts housing prices with *minimal error*.

## MLM, in code 

Just like in part 1, let's define our model. Since we have more than 1 weight, we need to use an array of `w`'s.

```c
struct mlinear_model {
	int num_weights;
	double *w;
	double b;
};
```


Now, you might have been able to think a few steps ahead, and realize that if we represent `w` as a vector, and `x` as a vector we can represent the output of the model as the dot product of $w$ and $x$, added with $b$

$$
\begin{align*}
\hat{y}
&= w_1 x_1 + \dots + w_nx_n + b \\\\
&= \vec{x} \cdot \vec{w} + b  \\\\
\end{align*}
$$


Ahh, much cleaner right? Well, not for long...

I'm going to take another leap here, and represent our $n$ length vector as a $(n \times 1)$ matrix. 

$$
w = \begin{bmatrix} w_1 \\\\ w_2 \\\\ w_3 \\\\ \vdots \\\\ w_n \end{bmatrix}
$$

You will see soon how this is helpful. Let's define our `matrix` struct.

```c
// matrix.h

// Just so we can swap out `double` for any float later
typedef double mfloat;

typedef struct {
	// row major
	mfloat *buf;
	int rows;
	int cols; 
} matrix;
```

Now, the model looks like

```c
// main.c
struct mlinear_model {
	matrix w;
	mfloat b;
};
```

> Notice that `num_weights` is now stored in the matrix. 

Since we switched to matrices, we need to use the matrix dot product instead of the vector dot product. I've added some notes that explain the code, in case you need to brush up on your linear algebra.

```c
// matrix.h

// Get an element in the matrix
mfloat
matrix_get(matrix m, int row, int col)
{
    return m.buf[row * m.cols + col];
}
// Set an element in the matrix
void
matrix_set(matrix m, int row, int col, mfloat val)
{
    m.buf[row * m.cols + col] = val;
}

// out = m1 dot m2
void
matrix_dot(matrix out, const matrix m1, const matrix m2)
{
	// On the ith row of the first matrix
    for (int row = 0; row < m1.rows; row++) {
		// On the jth column of the second matrix
        for (int col = 0; col < m2.cols; col++) {
			// Run the vector dot product and put the result in out[i][j]
            mfloat sum = 0.0;
			// m1.cols == m2.rows so k iterates over everything
            for (int k = 0; k < m1.cols; k++) {
                mfloat x1 = matrix_get(m1, row, k);
                mfloat x2 = matrix_get(m2, k, col);
                sum += x1 * x2;
            }
            matrix_set(out, row, col, sum);
        }
    }
}
```

First, let's note the following properties about the dot product between matrices $X$ and $W$

 1. It can only be computed *if and only if* `X.cols == W.rows`
 2. The resulting matrix has dimensions (`X.rows`, `W.cols`)

In this case,

$$
\begin{align*}
(1 \times n) &\cdot (n \times 1) &\to ~~~ &(1 \times 1) \\\\
X &\cdot W &\to ~~~&\text{result}\\\\
\end{align*}
$$

So, $X$ needs to be a *row vector* and $W$ needs to be a *column vector* so that we can replicate the behavior of the vector dot product.

$$
\begin{align*}
X \cdot W &= \begin{bmatrix} x_1 & x_2 & x_3 & \cdots & x_n \end{bmatrix} \cdot \begin{bmatrix} w_1 \\\\ w_2 \\\\ w_3 \\\\ \vdots \\\\ w_n \end{bmatrix} \\\\
&= x_1 w_1 + x_2 w_2 + \dots + x_nw_n
\end{align*}
$$

> You can think of the matrix dot product as taking each column in the second matrix, flipping it on its side, and dotting it with the row in the first matrix. 
> 
> Take a look at the code, and verify that the multiplication above will return the claimed result.

Now, we have the tools to code the prediction.

```c
// mlr.c

// x is a row vector, or (1 x n) matrix
mfloat predict(struct mlinear_model model, const matrix x) {
	// (1 x n) . (n x 1) => (1 x 1) matrix, which is a number
	mfloat result[1][1] = {0.0};
	matrix tmp = {.buf = result, .rows = 1, .cols = 1};
	// Set tmp to the result
	matrix_dot(tmp, x, model.w);
	return tmp.buf[0] + model.b;
}
```


## Optimizing the model

We are given 2 arrays: the input data, which is formatted as an $(m \times 4)$ matrix.

$$
X = \begin{bmatrix} x_1^1 & x_2^1 & x_3^1 & x_4^1 \\\\ x_1^2 & x_2^2 & x_3^2 & x_4^2 \\\\ \vdots & \vdots & \vdots & \vdots \\\\ x_1^m & x_2^m & x_3^m & x_4^m \end{bmatrix}
$$

Each row represents one sample, and the values in each column are the features described above.

And the corresponding house prices $Y$, in $1000s

$$
Y = \begin{bmatrix} y^1 \\\\ y^2 \\\\ y^3 \\\\ y^4 \\\\ \vdots \\\\ y^m \end{bmatrix}
$$
We're given 100 samples of the data, so $m = 100$. I'm going to store these in `data.h`

```c
// data.h

#define NUM_FEATURES 4
#define NUM_SAMPLES 100

static mfloat X_data[NUM_SAMPLES][NUM_FEATURES] = { /* data omitted*/ };

static mfloat Y_data[NUM_SAMPLES] = { /* data omitted */ };

static matrix X = {.buf = (mfloat *)X_data,
                   .rows = NUM_SAMPLES,
                   .cols = NUM_FEATURES};

static matrix Y = {
    .buf = (mfloat *)Y_data, .rows = NUM_SAMPLES, .cols = 1};

```

Now, it's optimization time!

This means we need to find the model with parameters $W$ and $b$ such that the error across all data samples is minimized. But what is our error? We can actually use the same definition from part 1, since we are still comparing two numbers $\hat{y}^{(i)}$ and $y^{(i)}$.

$$
\begin{align*}
J_{Wb}(X) =  \frac{1}{m} \sum_{i=0}^{m-1} (\hat{y}^{(i)} - y^{(i)})^2 \\\\
\end{align*}
$$

This is the *mean* of the *sum of squared differences*, or the average squared difference between the expected and actual $y$ values. The closer this value is to $0$, the better our model. Let's rewrite $\hat y$ in terms of $W, b,$  and $X$

$$
\begin{align*}
J_{Wb}(X) =  \frac{1}{m} \sum_{i=0}^{m-1} (X^{(i)}\cdot W + b - y^{(i)})^2 \\\\
\end{align*}
$$
Here, $X^{(i)}$ refers to the $i$th row of $X$. $X^{(i)}  \cdot W$ is the $i$th sample dotted with the weights. After we add the bias $b$, we get $\hat y$, or the prediction.

In code:

```c
// Return a single row matrix that represents the row of m at i
matrix matrix_get_row(matrix m, int i) {
  return (matrix){
      .buf = m.buf + i * m.cols, .rows = 1, .cols = m.cols};
}

mfloat cost(struct mlinear_model *model, matrix X,
                    matrix Y) {
  mfloat cost = 0.0;
  for (int i = 0; i < X.rows; i++) {
    matrix x_i = matrix_get_row(X, i);
    mfloat f_wb = predict(model, x_i);
    mfloat diff = matrix_get(Y, 0, i) - f_wb;
    cost += diff * diff;
  }
  return cost / (2.0 * X.rows);
}
```

#### How good was the guesstimate?

Let's go back to our guesstimate, and see how it fares. 

```c
int main() {
  matrix W = matrix_new(X.cols, 1);
  struct mlinear_model model = {.W = W, .b = 0.0};
  printf("Cost of zero model: %f\n", cost(&model, X, Y));
  matrix_set(W, 0, 0, 0.34);
  matrix_set(W, 1, 0, 3.3);
  matrix_set(W, 2, 0, 5.84);
  matrix_set(W, 3, 0, -8.07);
  model.b = 1634.5;
  printf("Cost of guesstimate model: %f\n",
         cost(&model, X, Y));
}
```

Output:

```
Cost of zero model:        3340683.781483
Cost of guesstimate model: 2641267.466911
```

Who would have known! It does slightly better than a guess. Back to gradient descent.

Let's bring in the calculus. We need to differentiate $J$ with respect to $W$ and $b$ so that we know which *direction*, and by what *magnitude*, we should move the weights and biases to reduce the error. 

$$
\begin{align\*}
\frac{\partial J}{\partial b}(X) 
&= \frac{2}{m} \sum\_{i=0}^{m-1} 
(X^{(i)} \cdot W + b - y^{(i)}) \\\\
&= \frac{2}{m} \sum\_{i=0}^{m-1} 
(\hat{y}^{(i)} - y^{(i)})
\end{align\*}
$$

Pretty simple, right? Not so much for $W$. Since $W$ is a matrix, each weight in $W$ has a separate effect on the output. This means we need to calculate a separate derivative for each weight $W^{(j)}$.

$$
\frac{\partial J}{\partial W^{(j)}} (X) = \frac{2}{m} \sum_{i=0}^{m-1} (X^{(i)} \cdot W + b - y^{(i)}) X^{(i)}_j
$$

> That $X^{(i)}_j$ term comes from differentiating $X^{(i)} \cdot W$. For a given weight $W^{(j)}$, and row $i$, the there would be a term that looks like $X^{(i)}_j W^{(j)}$. Since $X^{(i)}_j$ is a constant coefficient within the partial derivative with respect to $W^{(j)}$, it's going to get pulled out due to the chain rule.

Since we're already thinking in terms of matrices, let's write the derivative of $J$ with respect to the entire matrix $W$

$$
\begin{align\*}
\frac{\partial J}{\partial W} &= 
\begin{bmatrix}
\frac{\partial J}{\partial W^{(1)}} \\\\
\frac{\partial J}{\partial W^{(2)}} \\\\
\vdots \\\\
\frac{\partial J}{\partial W^{(n)}}
\end{bmatrix} =
\begin{bmatrix}
\frac{2}{m} \sum\_{i=0}^{m-1} (X^{(i)} \cdot W + b - y^{(i)}) X^{(i)}\_1 \\\\
\frac{2}{m} \sum\_{i=0}^{m-1} (X^{(i)} \cdot W + b - y^{(i)}) X^{(i)}\_2 \\\\
\vdots \\\\
\frac{2}{m} \sum\_{i=0}^{m-1} (X^{(i)} \cdot W + b - y^{(i)}) X^{(i)}\_n
\end{bmatrix} \\\\
&=
\frac{2}{m} \sum_{i=0}^{m-1} (X^{(i)} \cdot W + b - y^{(i)})
\begin{bmatrix}
X^{(i)}\_1 \\\\
X^{(i)}\_2 \\\\
\vdots \\\\
X^{(i)}\_n
\end{bmatrix} \\\\
&=
\frac{2}{m} \sum\_{i=0}^{m-1} (X^{(i)} \cdot W + b - y^{(i)}) \mathbf{X^{(i)}}\\\\
&= \frac{2}{m} \sum\_{i=0}^{m-1} (\hat{y}^{(i)} - y^{(i)}) \mathbf{X^{(i)}}
\end{align\*}
$$

That simplifies nicely! You may be starting to see why we love matrices in machine learning. It's a nice way to think about batch computations.

Now, for the code

```c
struct grad {
	matrix dJ_dW;
	mfloat dJ_db;
};


/* Compute gradient and write result to out. */
void compute_gradient(struct grad *out,
                      struct mlinear_model *model, const matrix X,
                      const matrix Y) {
  int m = X.rows;  // number of samples
  int n = X.cols;  // number of features

  // Using tmp to store each row of X
  matrix tmp = matrix_new(1, n);
  for (int i = 0; i < m; i++) {
    // tmp = X^(i)
    matrix curr_row = matrix_get_row(X, i);
    // y_hat = (X^(i) dot W) + b
    mfloat y_hat = predict(model, curr_row);
    // yi = y^(i)
    mfloat yi = matrix_get(Y, 0, i);
    // The term in parentheses
    mfloat err = y_hat - yi;

    /*
     * For dJ_dW, we need to multiply the error
     * by the current row, and add it to the running sum
     */

    // tmp = X^(i) * (y_hat^(i) - y^(i))
    matrix_scalar_mul(tmp, curr_row, err);
    // dJ_dW += tmp
    matrix_ip_T_add(out->dJ_dW, tmp);

    // dJ_db += (y_hat^(i) - y^(i))
    out->dJ_db += err;
  }

  /*
   * I'm going to replace 2/m with 1/m here since the 2
   * can be moved into alpha in the next step.
   */

  // dJ/db = (dJ/db) / m
  out->dJ_db /= m;
  // dJ/dW = (dJ/dW) / m
  matrix_scalar_ip_mul(out->dJ_dW, 1.0 / m);
  matrix_del(tmp);
}
```

> Note: `ip` in the matrix functions means the result is assigned to the first argument. Otherwise, the result is assigned to a buffer passed in as the first argument.
> 
> The new `matrix_*` functions are relatively simple, so I'm going to omit the code here to shorten the article. In fact, these functions were generated by `#define macros`, so the repo doesn't even contain the full source. Watch out for a Part 2.5 that goes over this!

I encourage you to go through the code and verify that it matches with the math. 

Now that we have the gradient, implementing gradient descent is straightforward. A refresher on the algorithm from part 1:

> Repeat until convergence:
> 1.  Set the initial weight & bias values: $W := [0~\dots~0]^T, b:=0$
> 2. Move the variables in the direction *opposite* to the derivatives,
>  with step size $\alpha$: $W := W - \alpha \frac{\partial J}{\partial W}, b:=b-\alpha \frac{\partial J}{\partial b}$
> 3. Go to step 2.

In code:

```c
void gradient_descent(struct mlinear_model *model, const matrix X,
                      const matrix Y, const int num_iterations,
                      const mfloat alpha) {
  // reusable buffer for gradient
  int n = X.cols, m = X.rows;
  matrix dJ_dW = matrix_new(n, 1);
  struct grad tmp_grad = {.dJ_dW = dJ_dW, .dJ_db = 0.0};

  for (int i = 0; i < num_iterations; i++) {
    // Log progress
    if (i % (num_iterations >> 4) == 0) {
      printf("\tCost at iteration %d: %f\n", i,
             compute_cost(model, X, Y));
    }
    // tmp_grad = current gradient for the model
    compute_gradient(&tmp_grad, model, X, Y);
    // dJ/dW *= -alpha
    matrix_scalar_ip_mul(tmp_grad.dJ_dW, -alpha);
    // W += dJ/dW
    matrix_ip_add(model->W, tmp_grad.dJ_dW);
    // b += -alpha * dJ/db
    model->b += -alpha * tmp_grad.dJ_db;
  }
  matrix_del(dJ_dW);
}
```

And that's all! Now we run the program on the data to see how our cost improves:

```c
int main() {
  // hyperparameters
  const int num_iterations = 1e7;
  const mfloat alpha = 1e-8;

  int n = X.cols, m = X.rows;
  matrix W = matrix_new(n, 1);
  struct mlinear_model model = {.W = W, .b = 0.0};

  printf("Initial cost: %f\n", compute_cost(&model, X, Y));
  gradient_descent(&model, X, Y, num_iterations, alpha);
  printf("Final cost: %f\n", compute_cost(&model, X, Y));
  printf("Model parameters:\n");
  matrix_print(model.W);
  printf(" b=%f\n", model.b);
}
```

Output:

```
Initial cost: 3340683.781483
        Cost at iteration 0: 3340683.781483
        Cost at iteration 625000: 161369.409722
        Cost at iteration 1250000: 161332.630253
        Cost at iteration 1875000: 161315.326515
        Cost at iteration 2500000: 161298.041198
        Cost at iteration 3125000: 161280.768143
        Cost at iteration 3750000: 161263.507342
        Cost at iteration 4375000: 161246.258784
        Cost at iteration 5000000: 161229.022462
        Cost at iteration 5625000: 161211.798366
        Cost at iteration 6250000: 161194.586488
        Cost at iteration 6875000: 161177.386819
        Cost at iteration 7500000: 161160.199351
        Cost at iteration 8125000: 161143.024075
        Cost at iteration 8750000: 161125.860983
        Cost at iteration 9375000: 161108.710065
Final cost: 161091.571313
Model parameters:
W=[ 2.1185e-01 ]
[ 5.8627e+00 ]
[ 4.5904e+00 ]
[ -1.1702e+01 ]

 b=5.310395
```

Awesome! We can see that on every iteration our cost is decreasing. The final model looks like

$$
\hat{y} = .212 x_1 + 5.86 x_2 + 4.59 x_3 - 11.7 x_4 + 5.31
$$
## Optimizing the optimizer

You may have noticed that even on the ten millionth iteration of gradient descent, the cost was still decreasing. Why is it so difficult to find the optimal parameters, especially when we only have 4 weights to optimize?

A large part is due to the difference in the spread of the features. Take a look at this box plot

{{< plotly src="X_box_plot.html" >}}

We can see that the weight for square footage needs to move *much more* than the weights for the other variables. However, the step size $\alpha$ is the same when we update each of the weights. This means that we will have to wait a long time after the other 3 feature weights have converged for the square footage weight to converge.

One solution is to choose $\alpha$ values proportional to the spread and move each weight by its corresponding $\alpha$. But this would mean storing another array of alpha values, which is expensive for large models.

A better solution is to actually modify our input data so that they have similar spreads. We can do this by fitting them to a normal distribution. First we calculate the mean of each feature.

$$
\mu^{(i)} = \frac{1}{n} \sum_{j=1}^{n} X^{(i)}_j
$$
Then, we calculate the spread, or standard deviation, of each feature.

$$
\sigma^{(i)} = \sqrt{\frac{\sum_{j=1}^{n}(X^{(i)}_j - \mu^{(i)})^2}{n}}
$$
And finally, we normalize it

$$
Z^{(i)} = \frac{X^{(i)} - \mu^{(i)}}{\sigma^{(i)}}
$$

Subtracting each column by its mean *centers* the data so that there is a similar spread on each side of the distribution. Dividing by $\sigma$ makes the spread of each column approximately the same. Let's write this in terms of matrix operations

```c
// Normalize the input data `X`
void z_score_normalize(matrix X) {
  int n = X.cols, m = X.rows;
  // Buffer for mu
  matrix mean = matrix_new(1, n);
  // Buffer for sigma
  matrix stdev = matrix_new(1, n);

  // Calculate mu
  for (int i = 0; i < NUM_SAMPLES; i++) {
    matrix_ip_add(mean, matrix_get_row(X, i));
  }
  matrix_scalar_ip_mul(mean, 1.0 / NUM_SAMPLES);

  // Calculate sigma
  matrix buf = matrix_new(1, n);
  for (int i = 0; i < NUM_SAMPLES; i++) {
    matrix row = matrix_get_row(X, i);
    matrix_sub(buf, mean, row);
    matrix_ip_square(buf);
    matrix_ip_add(stdev, buf);
  }
  matrix_ip_sqrt(stdev);

  // Calculate Z
  for (int i = 0; i < NUM_SAMPLES; i++) {
    matrix row = matrix_get_row(X, i);
    matrix_ip_sub(row, mean);
    matrix_ip_div(row, stdev);
  }
}
```

What does our normalized data look like?

{{< plotly src="X_normalized_box_plot.html" >}}

We can see the spreads are all on the same scale now. How does this improve our performance?

```c
int main() {
  int n = X.cols, m = X.rows;
  z_score_normalize(X);
  matrix W = matrix_new(n, 1);
  struct mlinear_model model = {.W = W, .b = 0.0};
  printf("Initial cost: %f\n", compute_cost(&model, X, Y));
  const int num_iterations = 1e7;
  const mfloat alpha = 1e-8;
  gradient_descent(&model, X, Y, num_iterations, alpha);
  printf("Final cost: %f\n", compute_cost(&model, X, Y));

  printf("Model parameters:\nW=");
  matrix_print(model.W);
  printf(" b=%f\n", model.b);
}
```

Output:

```
Initial cost: 3340683.781483
        Cost at iteration 0: 3340683.781483
        Cost at iteration 625000: 3305547.933747
        Cost at iteration 1250000: 3270852.031317
        Cost at iteration 1875000: 3236590.554988
        Cost at iteration 2500000: 3202758.054240
        Cost at iteration 3125000: 3169349.146943
        Cost at iteration 3750000: 3136358.518493
        Cost at iteration 4375000: 3103780.920969
        Cost at iteration 5000000: 3071611.172296
        Cost at iteration 5625000: 3039844.155415
        Cost at iteration 6250000: 3008474.817473
        Cost at iteration 6875000: 2977498.169013
        Cost at iteration 7500000: 2946909.283181
        Cost at iteration 8125000: 2916703.294939
        Cost at iteration 8750000: 2886875.400289
        Cost at iteration 9375000: 2857420.855511
Final cost: 2828334.976402
Model parameters:
W=[ 8.3053e+00 ]
[ 2.4385e+00 ]
[ 6.0186e+00 ]
[ -2.2844e+00 ]

 b=227.136534
```

It's even slower!? But wait—watch what happens when we change $\alpha = 10^{-8}$ to $\alpha = 1,$ which would have caused our old model to diverge.


```
Initial cost: 3340683.781483
        Cost at iteration 0: 3340683.781483
        Cost at iteration 625000: 136947.544865
        Cost at iteration 1250000: 136947.544865
        Cost at iteration 1875000: 136947.544865
        Cost at iteration 2500000: 136947.544865
        Cost at iteration 3125000: 136947.544865
        Cost at iteration 3750000: 136947.544865
        Cost at iteration 4375000: 136947.544865
        Cost at iteration 5000000: 136947.544865
        Cost at iteration 5625000: 136947.544865
        Cost at iteration 6250000: 136947.544865
        Cost at iteration 6875000: 136947.544865
        Cost at iteration 7500000: 136947.544865
        Cost at iteration 8125000: 136947.544865
        Cost at iteration 8750000: 136947.544865
        Cost at iteration 9375000: 136947.544865
Final cost: 136947.544865
Model parameters:
W=[ 8.1088e+03 ]
[ 8.1764e+02 ]
[ 6.6840e+02 ]
[ -3.6835e+03 ]

 b=2364.131545
```

Wow! Within the first log, it converged to the minimum value! In fact, with $\alpha=1$, we only need around $3 \times 10^4$ iterations for convergence instead of $10^7$! So we can see that using z-score normalization can speed up gradient descent by orders of magnitude, especially when the features have varying spreads. 

If you noticed that the new $W$ and $b$ are way off from the original ones,
that's because we fundamentally changed the input data. We are now modeling
based on the *magnitude of the deviation of a feature sample* instead of
its absolute value.

## Conclusion

So that's all for Multivariate Linear Regression. This is by far going to be the most difficult article in the series, so fear not of the future! All the code can be found [here](https://github.com/nathom/machine-learning-fundamentals/tree/main/part2). I encourage you to run it and modify its behavior.

If you have any questions or comments, feel free to leave a comment or shoot me an email.
