---
title: "Interactive MNIST Explorer"
date: 2024-02-20T11:53:54-07:00
draft: false
math: true
katex: true
comments: true
toc: true
summary: Draw digits on the canvas and watch an AI guess what it is!
---

{{< mnist >}}


In this article, we go over 3 basic models that tackle the MNIST dataset
in distinct ways and compare their properties, strengths, and weaknesses. You can
play with each one of these models above interactively and view their outputs in
the bar graphs.

## The Task

To those who aren't familiar with machine learning, converting an image into
a number might appear a daunting task. However, it becomes easier if we think
of the problem in the following way:

A grayscale picture is simply a grid of pixel brightnesses, which are real values. That is,
each image is some element in the set $\mathbb R^{w \times h}$, where $w,h$ are
the width and height of the image, respectively.  So, we can solve the problem if
we can find a function $f$ from $\mathbb R^{w \times h} \to \\{0, 1, \ldots, 8, 9\\}$.

To do this, we construct a model using our training images $x^{(n)}$ and labels $y^{(n)}$.

## Least Squares

This method involves creating 45 linear maps from $\mathbb R^{wh} \to \mathbb R$
for each unique pair $(i,j)$ selected from our categories 0..9 that infers
whether an image most likely belongs to pair $i$ or $j$. We can minimize Mean Squared Error (MSE)
using some linear algebra. First, instead of dealing with images in $\mathbb R^{w \times h}$, we can 
"flatten" them to $\mathbb R^{wh} \equiv \mathbb R^n$.

Define the weights of $(i,j)$ as $w_{ij}$, a vector of length $n+1$. To get the output of
the model, we compute

$$
\hat y_{ij} = \sum\_{k=1}^n w_{ij,k} x_{k} + w_{ij,n+1}
$$

where $x$ is a digit.

We want to minimize the MSE over all $m$ samples

$$
\begin{align*}
L_{ij} &= \frac{1}{n} \sum_{i=1}^m \left( \hat y^{(m)}\_{ij} - y_{ij}^{(m)} \right)^2 \\\\
&=\frac{1}{n} \sum_{i=1}^n \left( \begin{bmatrix} x_{ij}^\top & 1 \end{bmatrix} w_{ij}  - y_{ij}^{(n)} \right)^2
\end{align*}
$$

To do this, we create a new matrix $\mathbf X_{ij}$, which only contains images which
belong to class $i$ or $j$ along with a column of $\mathbf 1$  for the bias, and a matrix $y_{ij}$, which similarly only contains
labels in $\{i,j\}$, but replaces $i$ with $-1$ and $j$ with $1$.

Now, our problem has been reduced to 

$$
\min_{w_{ij}} || \mathbf X_{ij} \mathbf w_{ij} - \mathbf y_{ij}||_2^2
$$

The solution to this is given by $w_{ij} = \mathbf X_{ij}^\dagger y_{ij}$, where $\mathbf X^\dagger$ is the pseudoinverse of
the matrix $\mathbf X$ (Proof left as exercise to the reader 😁).

Once we have $w_{ij}$ for all $i,j$ pairs (45 in total),
we can represent our desired function $f$ as

```python
def f(x):
    score = [0] * 10
    for i, j, f_ij in pair_functions:
        out_ij = f_ij(x)
        if out_ij > 0:
            score[i] += 1
            score[j] -= 1
        else:
            score[j] += 1
            score[i] -= 1
    return argmax(score)
```

Each of the 45 models "votes" for either its $i$ or $j$.
The `score` array is what you see above in the bar graph.

## Fully Connected Network


A Fully Connected Network, or FCN is a much larger model than the least squares model. Instead of 
projecting our labels onto the principal subspace of the data, we can directly learn 
a mapping from the input space to the output space. 

For a single layer network, we assume that $f$ can be approximated by

$$
f(x) = g(\mathbf Ax)
$$

where $g$ is some nonlinear function. It is possible to learn the matrix $A$ such that
the error (Categorical Cross Entropy) is minimized in the local neighborhood through 
gradient descent. In the demo, we use a 2 layer network that maps the image to $\mathbb R^{128}$, and
the result of that to $\mathbb R^{10}$. This is represented by

$$
f(x) = h(\mathbf B(g(\mathbf Ax)))
$$

where we have to learn matrices $B \in \mathbb R^{10 \times 128} $ and $A \in \mathbb R^{128 \times n}$. In our case, $g(x)  = \max(0, x)$ and 

$$
h(z)_i = \frac{e^{z_i}}{\sum\_{j=1}^{10} e^{z\_j}}
$$


converts the output $\in \mathbb R^{10}$ to a probability distribution, which is shown above
in the bar graphs.

## Convolutional Network

A limitation of the two above models is that they don't see visual features like humans do.
For example, a handwritten `1` is a $1$ regardless of where it was painted on the canvas. However, since the
LS and FCN models do not have a notion of space or proximity, they will simply
point to the category which most likely have those exact pixels on.

Here, we bring in convolutions. Convolutions take an image and a *kernel*, run
the kernel through the image, and produce an output image that contain the weighted sum of the image pixels
and the kernel values.

<center>

{{< video src="convolution" autoplay="true" controls="false" loop="true" >}}

</center>

Notice how convolutions encode spatial data like plain networks do not. Since pixels nearby are usually 
highly correlated with each other, we can downsample the convolution output with a max pool and preserve
most of the information. After passing the image through a bunch of (trained) kernels, we get
a set of matrices that represent the existence of a learned *spatial feature*. Finally, we can flatten
and pass these into an FCN, which can now map spatial data into categories.

The output of this FCN (with softmax activation) is shown above.

## Model Comparison

Note: The last 3 columns are qualitative and relative to each other.

| Model                 | Number of Parameters           | Training Time | Inference Time | Accuracy  |
| ----------            | ------------------------------ | -----------   | ------------   | ------    |
| Least Squares         | $\text{35,325}$                | Low           | Fast           | Low       |
| FCN                   | $\text{101,760}$               | High          | Fast           | Good      |
| Convolutional Network | $\text{34,826}$                | Very High     | Slow           | Excellent |


Observations:

- Least Squares model is very fast but has a weak ability to generalize
- The CNN's parameters are very efficient to store
- Relative to the inference time of the CNN, LS and FCN are very fast

## Exercises

See how the models respond to these inputs:

- Empty canvas
- A `1` in the center
- A `1` in the far left side
- A `1` in the far right side
- A `0` with a line/dot in the center
- A `9`, with the top slightly disconnected
- Slightly rotated digits
- Very thin digits
- Very thick digits

Can you find 2 inputs that have a 1 pixel difference that map
to different categories?


## Implementation Details

All three of the models are running on your browser in plain JavaScript; no frameworks
or packages were used.

### Canvas

The $28 \times 28$ canvas is backed by an array of numbers that contain
the alpha value that is displayed. Every time any pixel is updated, the entire
thing is redrawn. The only other interesting detail is the brightness dropoff function
I used:

```javascript
const plateau = 0.3;
// dist is distance^2 from center
const alpha = Math.min(1 - dist / r2 + plateau, 1);
pixels[yc * 28 + xc] = Math.max(pixels[yc * 28 + xc], alpha);
```

I initially tried a dropoff of `1-dist/r2` but it faded
the center too much. So I added the `plateau` variable that
shifts the function up but clamped it down with `Math.min` so
that alpha doesn't exceed 1. This gives the brush a more natural look. 


### Least Squares

I got the weights from a project I did in ECE 174 with Professor Piya Pal. Inference
is simply 45 dot products and scoring

```javascript
function evalLSModel(digit, weights) {
    const scores = new Array(10).fill(0);
    for (const pairConfig of weights) {
        const [i, j, w] = pairConfig;
        // Vector dot product
        const result = vdot(digit, w);
        if (result > 0) {
            scores[i] += 1;
            scores[j] -= 1;
        } else {
            scores[j] += 1;
            scores[i] -= 1;
        }
    }
    return scores;
}
```

### Fully Connected Network


The main work with FCN inference is the matrix dot product,
which I implemented in the standard way.


```javascript
function matrixDot(matrix1, matrix2, rows1, cols1, rows2, cols2) {
    // Check if the matrices can be multiplied
    if (cols1 !== rows2) {
        console.error("Invalid matrix dimensions for dot product");
        return null;
    }

    // Initialize result matrix with zeros
    const result = new Array(rows1 * cols2).fill(0);

    // Perform dot product
    for (let i = 0; i < rows1; i++) {
        for (let j = 0; j < cols2; j++) {
            for (let k = 0; k < cols1; k++) {
                result[i * cols2 + j] +=
                    matrix1[i * cols1 + k] * matrix2[k * cols2 + j];
            }
        }
    }

    return result;
}
```

I stored matrices in a single 1D `Array` for better cache locality
and fewer heap allocations. As per the formula above, inference
consists of 2 matrix dot products and 2 activation function applications.
The `push(1)` calls are to calculate the bias.

```javascript
function evalNN(digit, weights) {
    const digitCopy = [...digit];
    digitCopy.push(1);
    // layer 1 params
    const [w1, [rows1, cols1]] = weights[0];
    const out1 = matrixDot(digitCopy, w1, 1, digitCopy.length, rows1, cols1).map(relu);
    const [w2, [rows2, cols2]] = weights[1];
    out1.push(1);
    const out2 = matrixDot(out1, w2, 1, out1.length, rows2, cols2);
    return softmax(out2);
}
```

### Convolutional Network

The convnet here is quite small. In Pytorch it is

```python
nn.Sequential(
    nn.Conv2d(1, 32, kernel_size=3),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(32, 64, kernel_size=3),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Dropout(0.5),
    nn.Linear(1600, 10),
    nn.Softmax(dim=1)
)
```

For inference, we just have to port the forward passes to JavaScript.
Conv2d (with in/out channels) is given by

```javascript
function conv2d(
    nInChan,
    nOutChan,
    inputData,
    inputHeight,
    inputWidth,
    kernel,
    bias,
) {
    if (inputData.length !== inputHeight * inputWidth * nInChan) {
        console.error("Invalid input size");
        return;
    }
    if (kernel.length !== 3 * 3 * nInChan * nOutChan) {
        console.error("Invalid kernel size");
        return;
    }

    const kernelHeight = 3;
    const kernelWidth = 3;

    // Compute output dimensions
    const outputHeight = inputHeight - kernelHeight + 1;
    const outputWidth = inputWidth - kernelWidth + 1;

    const output = new Array(nOutChan * outputHeight * outputWidth).fill(0);

    for (let i = 0; i < outputHeight; i++) {
        for (let j = 0; j < outputWidth; j++) {
            for (let outChan = 0; outChan < nOutChan; outChan++) {
                let sum = 0;
                // apply filter at single location over all input channels
                for (let inChan = 0; inChan < nInChan; inChan++) {
                    for (let row = 0; row < 3; row++) {
                        for (let col = 0; col < 3; col++) {
                            const inI =
                                inChan * (inputHeight * inputWidth) +
                                (i + row) * inputWidth +
                                (j + col);

                            const kI =
                                outChan * (nInChan * 3 * 3) +
                                inChan * (3 * 3) +
                                row * 3 +
                                col;
                            sum += inputData[inI] * kernel[kI];
                        }
                    }
                }
                sum += bias[outChan];
                const outI =
                    outChan * (outputHeight * outputWidth) +
                    i * outputWidth +
                    j;
                output[outI] = sum;
            }
        }
    }
    return output;
}
```

I know it's ugly. I'm just putting it here for reference. Heads up for maxpool:

```javascript
function maxPool2d(nInChannels, inputData, inputHeight, inputWidth) {
    if (inputData.length !== inputHeight * inputWidth * nInChannels) {
        console.error("maxpool2d: invalid input height/width");
        return;
    }
    const poolSize = 2;
    const stride = 2;
    const outputHeight = Math.floor((inputHeight - poolSize) / stride) + 1;
    const outputWidth = Math.floor((inputWidth - poolSize) / stride) + 1;
    const output = new Array(outputHeight * outputWidth * nInChannels).fill(0);

    for (let chan = 0; chan < nInChannels; chan++) {
        for (let i = 0; i < outputHeight; i++) {
            for (let j = 0; j < outputWidth; j++) {
                let m = 0;
                for (let row = 0; row < poolSize; row++) {
                    for (let col = 0; col < poolSize; col++) {
                        const ind =
                            chan * (inputHeight * inputWidth) +
                            (i * stride + row) * inputWidth +
                            (j * stride + col);
                        m = Math.max(m, inputData[ind]);
                    }
                }
                const outI =
                    chan * (outputHeight * outputWidth) + i * outputWidth + j;
                output[outI] = m;
            }
        }
    }
    return output;
}
```

And yes, the only reason I am dealing with that abomination of index calculating code
is for that **Blazing fast 🔥JavaScript🔥 Web App** performance. And finally,
here's the function that ties it all together:

```javascript
function evalConv(digit, weights) {
    const [
        [f1, fshape1], // conv filter weights
        [b1, bshape1], // conv bias
        [f2, fshape2],
        [b2, fbshape2],
        [w, wshape],   // fcn weights
        [b, bshape],   // fcn bias
    ] = weights;

    const x1 = conv2d(1, 32, digit, 28, 28, f1, b1).map(relu);
    const x2 = maxPool2d(32, x1, 26, 26);
    const x3 = conv2d(32, 64, x2, 13, 13, f2, b2).map(relu);
    const x4 = maxPool2d(64, x3, 11, 11);
    const x5 = matrixDot(w, x4, 10, 1600, 1600, 1);
    const x6 = vsum(x5, b);
    const out = softmax(x6);
    return out;
}
```

## Conclusion

I hope you all enjoy playing around with the app. If you have any questions
or feedback, feel free to leave a comment below.
