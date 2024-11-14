---
title: The Mechanics of Causal Self Attention 
date: 2024-11-13T14:51:11-08:00
draft: false
math: true
comments: true
toc: true
---

## Begin

Causal self-attention is the mechanism underpinning most of the advances in AI since 2017. In this article, I will step through the computation and hopefully gain a better intuition of how it works.

$$
\\text{SelfAttention}(\\mathbf{Q}, \\mathbf{K}, \\mathbf{V}) = \\text{softmax}\\left( \\text{mask} \\left(\\frac{\\mathbf{Q} \\mathbf{K}^T}{\\sqrt{d}}\\right) \\right) \\mathbf{V}
$$

At a high level, this function takes one *sequence* and transforms it into another. A sequence is a list of token embeddings, a tensor of shape $L \\times d$, where $L$ is the input sequence length and $d$ is the embedding dimension. Each row of this matrix corresponds to one input token, which is represented as a $d$-dimensional vector.

So why then, are there 3 inputs to $\\text{SelfAttention}$? This is because, in the Transformer architecture, the input sequence is projected by 3 different $d \\times d$ linear layers. If $\\mathbf{X}$ is the input sequence,

$$
\\mathbf{Q} = \\mathbf{X}\\mathbf{W\_Q}, \\mathbf{K} = \\mathbf{X}\\mathbf{W\_K}, \\mathbf{V} = \\mathbf{X}\\mathbf{W\_V}
$$

where $\\mathbf{W}$ are $d \\times d$. So, $\\mathbf{Q},\\mathbf{K},\\mathbf{V}$ are simply different representations of the same input sequence.

Let's compute $\\text{SelfAttention}$ step-by-step. First, we do $\\mathbf{Q}\\mathbf{K}^T$, which is a $L \\times d$ by $d \\times L$ dot product, resulting in an $L \\times L$ output. What does this do?

$$
\\begin{align*}
\\mathbf{Q} \\mathbf{K}^T = \\begin{bmatrix} \\mathbf{q}\_1 \\\\ \\mathbf{q}\_2 \\\\ \\vdots \\\\ \\mathbf{q}\_L \\end{bmatrix} \\begin{bmatrix} \\mathbf{k}\_1^T & \\mathbf{k}\_2^T & \\cdots & \\mathbf{k}\_L^T \\end{bmatrix}
= \\begin{bmatrix} 
\\mathbf{q}\_1 \\mathbf{k}\_1^T & \\mathbf{q}\_1 \\mathbf{k}\_2^T & \\cdots & \\mathbf{q}\_1 \\mathbf{k}\_L^T \\\\ 
\\mathbf{q}\_2 \\mathbf{k}\_1^T & \\mathbf{q}\_2 \\mathbf{k}\_2^T & \\cdots & \\mathbf{q}\_2 \\mathbf{k}\_L^T \\\\ 
\\vdots & \\vdots & \\ddots & \\vdots \\\\ 
\\mathbf{q}\_L \\mathbf{k}\_1^T & \\mathbf{q}\_L \\mathbf{k}\_2^T & \\cdots & \\mathbf{q}\_L \\mathbf{k}\_L^T 
\\end{bmatrix}
\\end{align*}
$$

The result of $\\mathbf{q}\_i \\mathbf{k}^T\_j$ is a scalar ($1 \\times d$ dot $d \\times 1$), and it is the vector dot-product between $\\mathbf{q}\_i$ and $\\mathbf{k}\_j$. If we remember the formula

$$
\\mathbf{a} \\cdot \\mathbf{b} = \\|\\mathbf{a}\\| \\|\\mathbf{b}\\| \\cos \\theta
$$

we see that the dot-product is positive when $\\theta$, the angle between $\\mathbf{a}$ and $\\mathbf{b}$, is close to 0º and negative when the angle is 180º, or when they point in opposite directions. We can interpret the dot product as a similarity metric, where positive values indicate similar vectors, and negative values indicate the opposite.

So our final $L \\times L$ matrix is filled with similarity scores between every pair of $\\mathbf{q}$ and $\\mathbf{k}$ tokens. The result is divided by $\\sqrt{d}$ to prevent the variance from exploding for large embedding dimensions. See [Appendix](#why-scale-by-sqrtd) for details.

The next step is to apply the $\\text{mask}$ function, which sets all values that are not in the lower-triangular section of the input matrix to $-\\infty$.

$$
\\text{mask}\\left(\\frac{1}{\\sqrt{d}} \\mathbf{Q}\\mathbf{K}^T\\right) = \\frac{1}{\\sqrt{d}} \\begin{bmatrix}
\\mathbf{q}\_1 \\mathbf{k}\_1^T & -\\infty & -\\infty & \\cdots & -\\infty \\\\
\\mathbf{q}\_2 \\mathbf{k}\_1^T & \\mathbf{q}\_2 \\mathbf{k}\_2^T & -\\infty & \\cdots & -\\infty \\\\
\\mathbf{q}\_3 \\mathbf{k}\_1^T & \\mathbf{q}\_3 \\mathbf{k}\_2^T & \\mathbf{q}\_3 \\mathbf{k}\_3^T & \\cdots & -\\infty \\\\
\\vdots & \\vdots & \\vdots & \\ddots & \\vdots \\\\
\\mathbf{q}\_L \\mathbf{k}\_1^T & \\mathbf{q}\_L \\mathbf{k}\_2^T & \\mathbf{q}\_L \\mathbf{k}\_3^T & \\cdots & \\mathbf{q}\_L \\mathbf{k}\_L^T 
\\end{bmatrix}
$$

To this, we apply $\\text{softmax}$, which converts each row of values in the matrix into a probability distribution. The function is defined as a mapping from $\\mathbb R^L \\to \\mathbb R^L$, where the $i$th output element is given by

$$
\\text{softmax}(\\mathbf{x})\_i = \\frac{e^{x\_i}}{\\sum\_{j=1}^L e^{x\_j}} \\quad \\text{for } i = 1, 2, \\ldots, L
$$

Two things to note here:

1. The sum of all output elements is $1$, as is expected for a probability distribution
2. If an input element $x\_i$ is $-\\infty$, then $\\text{softmax}(x)\_i = 0$

After applying the $\\text{softmax}$ function to the masked similarity scores, we obtain:

$$
\\mathbf{S} = \\text{softmax}\\left(\\text{mask}\\left(\\frac{1}{\\sqrt{d}} \\mathbf{Q} \\mathbf{K}^T\\right)\\right) = \\begin{bmatrix}
S\_{1,1} & 0 & 0 & \\cdots & 0 \\\\
S\_{2,1} & S\_{2,2} & 0 & \\cdots & 0 \\\\
S\_{3,1} & S\_{3,2} & S\_{3,3} & \\cdots & 0 \\\\
\\vdots & \\vdots & \\vdots & \\ddots & \\vdots \\\\
S\_{L,1} & S\_{L,2} & S\_{L,3} & \\cdots & S\_{L,L}
\\end{bmatrix}
$$

Where the entries  $S\_{i,j}$  are defined as:

$$
S\_{i,j} = \\frac{e^{\\text{mask}\\left(\\frac{\\mathbf{Q} \\mathbf{K}^T}{\\sqrt{d}}\\right)\_{i,j}}}{\\sum\_{k=1}^{L} e^{\\text{mask}\\left(\\frac{\\mathbf{Q} \\mathbf{K}^T}{\\sqrt{d}}\\right)\_{i,k}}}
$$

The resulting matrix $\\mathbf{S}$ has probability distribution rows of length $L$. The final step is to map our value matrix $\\mathbf{V}$ by these probability distributions to give us our new sequence.

$$
\\begin{align*}
\\text{SelfAttention}(\\mathbf{Q},\\mathbf{K},\\mathbf{V}) &= \\mathbf{S}\\mathbf{V} \\\\
&=  \\begin{bmatrix}
S\_{1,1} & 0 & 0 & \\cdots & 0 \\\\
S\_{2,1} & S\_{2,2} & 0 & \\cdots & 0 \\\\
S\_{3,1} & S\_{3,2} & S\_{3,3} & \\cdots & 0 \\\\
\\vdots & \\vdots & \\vdots & \\ddots & \\vdots \\\\
S\_{L,1} & S\_{L,2} & S\_{L,3} & \\cdots & S\_{L,L}
\\end{bmatrix} \\begin{bmatrix}
\\mathbf{V}\_1 \\\\
\\mathbf{V}\_2 \\\\
\\mathbf{V}\_3 \\\\
\\vdots \\\\
\\mathbf{V}\_L
\\end{bmatrix}  \\\\
&= \\begin{bmatrix}
\\begin{array}{l}
S\_{1,1} \\mathbf{V}\_1 \\\\
S\_{2,1} \\mathbf{V}\_1 + S\_{2,2} \\mathbf{V}\_2 \\\\
S\_{3,1} \\mathbf{V}\_1 + S\_{3,2} \\mathbf{V}\_2 + S\_{3,3} \\mathbf{V}\_3 \\\\
\\hspace{3.cm} \\vdots \\\\
S\_{L,1} \\mathbf{V}\_1 + S\_{L,2} \\mathbf{V}\_2 + \\cdots + S\_{L,L} \\mathbf{V}\_L \\\\
\\end{array}
\\end{bmatrix}
\\end{align*}
$$

Note that $S\_{i,j}$ is a scalar, and $\\mathbf{V}\_k$ is a $1 \\times d$ embedding vector. Visually, we observe that SelfAttention is selectively combining Value tokens, weighted by a probability distribution generated by how well the queries and keys attend to each other, i.e. have a large inner product. We also see the weight of an output token at index $i$ is solely generated by input tokens with index $\\le i$, due to the *causal mask* we applied earlier. This is based on the *causal assumption*, that the an output token $\\mathbf{O}\_i$ does not depend on future tokens, which is required when training *autoregressive* (i.e. next token prediction) models.

Hopefully you found this helpful!

## Appendix

### Why Scale by $\\sqrt{d}$?


Assume that $\\mathbf{q}\_i, \\mathbf{k}\_i \\sim \\mathcal{N}(\\mu = 0, \\sigma^2 = 1)$ and i.i.d. Let's compute the mean and variance of $s = \\mathbf{q} \\cdot \\mathbf{k}$.

The mean is trivially zero:

$$
\\mathbb{E}[s] = \\mathbb{E}\\left[ \\sum\_{i=1}^d \\mathbf{q}\_i \\mathbf{k}\_i \\right] = \\sum\_{i=1}^d \\mathbb{E}[\\mathbf{q}\_i \\mathbf{k}\_i] = \\sum\_{i=1}^d \\mathbb{E}[\\mathbf{q}\_i] \\mathbb{E}[ \\mathbf{k}\_i] = 0
$$

$$
\\text{Var}(s) = \\mathbb{E}[s^2] - (\\mathbb{E}[s])^2 = \\mathbb{E}[s^2] = d
$$

because 

$$
\\mathbb{E}[s^2] = \\mathbb{E}\\left[ \\sum\_{i=1}^d \\sum\_{j=1}^d \\mathbf{q}\_i \\mathbf{k}\_i \\mathbf{q}\_j \\mathbf{k}\_j \\right] = \\sum\_{i=1}^d \\sum\_{j=1}^d \\mathbb{E}[\\mathbf{q}\_i \\mathbf{k}\_i \\mathbf{q}\_j \\mathbf{k}\_j]
$$

which is $0$ for $i \\ne j$ (since $\\mathbf{q}\_i,\\mathbf{q}\_j$ and $\\mathbf{k}\_i,\\mathbf{k}\_j$ are i.i.d). For $i=j$, 

$$
\\sum\_{i=1}^d \\mathbb{E}[\\mathbf{q}\_i^2 \\mathbf{k}\_i^2] = \\sum\_{i=1}^d \\mathbb{E}[\\mathbf{q}\_i^2] \\mathbb{E}[\\mathbf{k}\_i^2] = \\sum\_{i=1}^d 1 \\cdot 1 = d
$$

since $\\mathbb{E}[\\mathbf{q}\_i^2] = \\mathbb{E}[\\mathbf{k}\_i^2] = \\sigma^2 = 1$. 

So if we scale by $1/\\sqrt{d}$, our new variance is 

$$
\\text{Var}(\\frac{s}{\\sqrt{d}}) =\\text{Var}(\\frac{\\mathbf{q} \\cdot \\mathbf{k}}{\\sqrt{d}}) = \\frac{1}{d} \\text{Var}(\\mathbf{q} \\cdot \\mathbf{k}) = 1
$$

as desired.

### Multi-Head Attention

Most modern systems use multi-head attention, which computes $\\text{SelfAttention}$ in parallel over several "heads". We usually let $d\_k=d\_v= d\_{\\text{model}} / H$, where $H$ is the number of heads.

$$
\\begin{aligned}
\\mathbf{Q}\_h &= \\mathbf{X} \\mathbf{W}^Q\_h \\quad &\\mathbf{W}^Q\_h \\in \\mathbb{R}^{d\_{\\text{model}} \\times d\_k} \\\\
\\mathbf{K}\_h &= \\mathbf{X} \\mathbf{W}^K\_h \\quad &\\mathbf{W}^K\_h \\in \\mathbb{R}^{d\_{\\text{model}} \\times d\_k} \\\\
\\mathbf{V}\_h &= \\mathbf{X} \\mathbf{W}^V\_h \\quad &\\mathbf{W}^V\_h \\in \\mathbb{R}^{d\_{\\text{model}} \\times d\_v}
\\end{aligned}
$$

$$
\\text{head}\_h = \\text{SelfAttention}(\\mathbf{Q}\_h, \\mathbf{K}\_h, \\mathbf{V}\_h) = \\text{softmax}\\left( \\text{mask} \\left( \\frac{\\mathbf{Q}\_h \\mathbf{K}\_h^T}{\\sqrt{d\_k}} \\right) \\right) \\mathbf{V}\_h
$$

$$
\\begin{aligned}
\\text{MultiHead}(\\mathbf{Q}, \\mathbf{K}, \\mathbf{V}) &= \\text{Concat}(\\text{head}\_1, \\text{head}\_2, \\ldots, \\text{head}\_H)
\\end{aligned}
$$
