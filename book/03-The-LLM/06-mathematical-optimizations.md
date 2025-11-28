# Mathematical optimizations

I mentioned way back [in the introduction](#conceptual-layers) that I find it useful to think about LLMs first in terms of the fundamental concepts, and then in terms of the mathematical optimizations of those concepts. Until now, I've been focusing exclusively on the conceptual layers. In this chapter, I'll describe how those get bundled into mathematical objects that are more efficient to compute.

There are two major components to this:

- Turning vectors-of-vectors into matrices
- Increasing the rank of all tensors by 1, so that we can add a batching dimension

:::{note} Assume $d = \delta$
In this chapter, I'll be relying on the common constraint that each transformation's input and output dimensions are the same. I mentioned that this is a common model constraint in the chapter on attention, and the previous chapter provided additional motivation for it in the context of residual connections. Since it also simplifies this chapter's math, I'll just assume it from here on out.
:::

## The architecture's conceptual shape

Before we dive into the mathematical optimizations, let's take a look at the LLM's architecture once more, this time focusing on the shapes of the learned parameters and activations.

I'll skip the tokenization phase, since that's effectively a preparation step that happens before the LLM itself runs.

{drawio}`An overview of the LLM architecture, showing n vectors of size d for most of the flow, and a final output of n vectors of size v|images/tensors/architecture-concepts`

For most of the LLM, the activations are in the form of $n$ vectors, each size $d$. The final output is still $n$ vectors, but each sized $v$ (the vocabulary size).

## Vectors of vectors → matrices

The basic "lifting" we'll do is to to turn vectors of vectors into matrices. This will let us turn "for each outer vector, do some stuff" loops into matrix multiplication. This doesn't change what's going on conceptually, but it lets us do the math on GPUs and TPUs that process it much more quickly.

All we need to do is turn each "outer" vector into a row in a matrix:

$$
\begin{array}{llll}
[\; 1.32, & 5.91, & 5.71, & \dots \;] \\
[\; 6.16, & 4.81, & 3.62, & \dots \;] \\
[\; 8.27, & 9.53, & 2.44, & \dots \;] \\
[\; \dots, & \dots, & \dots, & \dots \;]
\end{array}
\quad \Longrightarrow \quad
\begin{bmatrix}
1.32 & 5.91 & 5.71 & \dots \\
6.16 & 4.81 & 3.62 & \dots \\
8.27 & 9.53 & 2.44 & \dots \\
\dots & \dots & \dots & \dots
\end{bmatrix}
$$

When we apply this to the right-hand column of the diagram above, it turns "$n$ vectors of size $d$" into "a single $n \times d$ matrix".

Let's work through what that means for the calculations I've described in the previous chapters.

### Calculating attention

Recall that for a single query embedding token $t_q$, we calculated attention by:

1. For each input embedding, calculating its query vector $q = t_q \cdot W_q$ (size $d$). Then:
    1. For every input embedding $t_i$ (there are $n$ of them), calculate an attention score:
        1. Calculate the key vector $k = t_i \cdot W_k$ (size $d$)
        2. Calculate the dot product $q \cdot k$ to get the attention score (a scalar)
    2. Treat those $n$ attention scores as a vector; scale and softmax that to get the attention weight vector (size $n$)
    3. For every input embedding $t_i$, calculate a value vector $v = t_i \cdot W_v$ (there are $n$ such vectors, each size $d$)
    4. For each value vector $v$ (there are $n$ of them), multiply that $d$-sized value vector by the corresponding attention weight (a scalar). This gives you $n$ $d$-sized vectors.
    5. Sum them to get the context vector (size $d$)

Let's see how much of this we can turn into matrix math. (Spoiler alert: all of it.)

:::{warning} Warning! Math!
This part is a bit dense, sorry. Make sure you understand @matrix-math-details from the background chapter on matrix math.
:::

#### Calculating the key matrix $K$

I'll start with step 1.1.1. Let's calculate the key vector $k_i$ for input embedding $t_i$. I'll be using $model$ as the embedding dimension size, which is common nomenclature. (The math in this section is per head, so typically $d = \frac{d_{model}}{head}$.)

$$
\begin{align}
k_i & = t_i \cdot W_k \\
    & = \begin{bmatrix} t_{i,1} & t_{i,2} & \dots \end{bmatrix} \cdot W_k \\
    & = \underbrace{\begin{bmatrix} t_{i,0} & t_{i,1} & \dots \end{bmatrix}}_{1 \times d_{model}} \cdot \underbrace{\begin{bmatrix} w_{0,0} & w_{0,1} & \dots \\ w_{1,0} & w_{1,1} & \dots \\ \vdots & \vdots & \ddots \end{bmatrix}}_{d_{model} \times d} \\
    & = \begin{bmatrix}
          \begin{bmatrix} t_{i,0} & t_{i,1} & \dots \end{bmatrix}
          \cdot
          \begin{bmatrix} w_{0,0} \\ w_{1,0} \\ \dots \end{bmatrix}
          \quad
        &
          \begin{bmatrix} t_{i,0} & t_{i,1} & \dots \end{bmatrix}
            \cdot
          \begin{bmatrix} w_{0,1} \\ w_{1,1} \\ \dots \end{bmatrix}
          \quad
        & \dots
      \end{bmatrix}
\end{align}
$$

As you can see, $k_i$ is a vectors that's the result of a $d_{model}$ vector (which is equivalent to a $a \times d_{model}$ matrix) multiplied by a $d_{model} \times d$ matrix, in this case $W_k$.

If we do this for each embedding, we get a matrix that we'll call $K$:

$$
\begin{align}
K & = \left. \begin{bmatrix}
        t_0 \cdot W_k \\
        t_1 \cdot W_k \\
        \vdots
      \end{bmatrix} \right\} n \text{ elements} \\[3.5em]
  & = \begin{bmatrix}
        \begin{bmatrix} t_{0,0} & t_{0,1} & \dots \end{bmatrix}
        \cdot
        \begin{bmatrix} w_{0,0} \\ w_{1,0} \\ \dots \end{bmatrix}
        \quad
      &
        \begin{bmatrix} t_{0,0} & t_{0,1} & \dots \end{bmatrix}
          \cdot
        \begin{bmatrix} w_{0,1} \\ w_{1,1} \\ \dots \end{bmatrix}
        \quad
      & \dots
      \\[2.5em]
        \begin{bmatrix} t_{1,0} & t_{1,1} & \dots \end{bmatrix}
        \cdot
        \begin{bmatrix} w_{0,0} \\ w_{1,0} \\ \dots \end{bmatrix}
        \quad
      &
        \begin{bmatrix} t_{1,0} & t_{1,1} & \dots \end{bmatrix}
          \cdot
        \begin{bmatrix} w_{0,1} \\ w_{1,1} \\ \dots \end{bmatrix}
        \quad
      & \dots
      \\ \vdots & \vdots &\ddots
    \end{bmatrix} \\[3.5em]
  & = \underbrace{X \cdot W_k}_{n \times d}
\end{align}
$$

Each row in this matrix corresponds to an input embedding's key vector.

#### Calculating attention scores matrix

Now, we can move onto the raw attention scores. This corresponds to step 1.1.2 above.

First, let's calculate the query matrix $Q$. This is exactly the same as the key matrix $K$, except that it uses $W_q$ instead of $W_k$. Because the progression from vectors-of-vectors to matrix is the same, I won't spell it out in full.

$$
Q = \underbrace{X \cdot W_q}_{n \times d}
$$

We want the attention scores as a matrix, with each row containing one token's scores. To do this, we'll start with $Q$, and for each of its rows, we want the result's columns to be that row dot-producted with each key vector:

$$
\text{attention scores} = \begin{bmatrix}
  (Q_0 \cdot \text{(key vector 0)}) & (Q_0 \cdot \text{(key vector 1)}) & \dots \\
  (Q_1 \cdot \text{(key vector 0)}) & (Q_1 \cdot \text{(key vector 1)}) & \dots \\
  \vdots & \vdots & \ddots
\end{bmatrix}
$$

We _almost_ have this: the only problem is that our $K$ matrix has each key vector as a row, and we need them as columns. In other words, we need to [transpose](#matrix-transposition) it:

$$
\begin{align}
\text{attention scores} & = Q \cdot K^T \\
 & = \underbrace{QK^T}_{n \times n} \; \text{(as it's more commonly written)}
\end{align}
$$

(scale-and-softmax-matrix)=

#### Scale and softmax

Next, we just need to scale each element in the attention scores by dividing it by $\sqrt{d}$, and then apply softmax. This corresponds to step 2 above.

$$
\text{attention weights} = A = \text{softmax}\left( \frac{QK^T}{\sqrt{d}} \right)
$$

Neither of these changes the dimensions of the matrix, so it's still $n \times n$.

#### Applying values to get attention

Finally, we'll apply our weights against the value vectors, and sum the results. This corresponds to steps 3 - 4 above.

First, we'll get the value matrix $V$, similar to the above. This is step 3.

$$
V = \underbrace{X \cdot W_v}_{n \times d}
$$

Each row in this matrix is one value vector.

Before we go further, let's step back and try to compute just a single context vector (that is, just a single token's attentions) with what we have. This means that within the context of a single query token $Q_i$, we want to:

- take all the value vectors:

  $$
  \left.\begin{bmatrix} V_0 \\ V_1 \\ \vdots \end{bmatrix}\right\} \text{$n$ vectors, each size $d$}
  $$
- multiply each one by the corresponding attention weights for this query token:

  $$
  \begin{bmatrix}A_{i,0}V_0 \\ A_{i,1}V_1 \\ \vdots \end{bmatrix}
  = \left.
    \underbrace{
      \begin{bmatrix}
      A_{i,0}V_{0,0} & A_{i,0}V_{0,1} & \dots \\
      A_{i,1}V_{1,0} & A_{i,1}V_{1,1} & \dots \\
      \vdots & \vdots & \ddots
      \end{bmatrix}
    }_{d}
    \right\} n
  $$
- sum the $n$ vectors to get a single vector, size $d$

  $$
  \begin{bmatrix}
  (A_{i,0}V_{0,0} + A_{i,1}V_{1,0} + \dots)
  & (A_{i,0}V_{0,1} + A_{i,1}V_{1,1} + \dots)
  & \dots
  \end{bmatrix}
  $$

Now that we have the context vector for a given query vector $Q_i$, let's see what they'd look like stacked as rows of a matrix:

$$
\begin{bmatrix}
(A_{0,0}V_{0,0} + A_{0,1}V_{1,0} + \dots) & (A_{0,0}V_{0,1} + A_{0,1}V_{1,1} + \dots) & \dots
\\[1em]
(A_{1,0}V_{0,0} + A_{1,1}V_{1,0} + \dots) & (A_{1,0}V_{0,1} + A_{1,1}V_{1,1} + \dots) & \dots
\\[1em]
\vdots & \vdots & \ddots
\end{bmatrix}
$$

This may look familiar: it's just the matrix multiplication $AV$.

#### Putting it together

If we substitute $A$ in the expression above with the expression from @scale-and-softmax-matrix above, we get:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d}} \right)V
$$

This is the canonical representation of attention, and is somewhat famous within the literature of LLMs.

This means we can calculate the attention for a head using pretty much all matrix math:

- calculate $Q$, $K$, and $V$ as $XW_q$, $XW_k$, and $XW_v$ respectively
- plug them into the $\text{Attention}(Q, K, V)$ function above

#### Multi-head attention

:::{warning} TODO
:::

### FFNs

:::{warning} TODO
:::

### Normalization

:::{warning} TODO
:::

## Batching

Up until now, we've been working with one input at a time. In practice, GPUs and especially TPUs can process multiple inputs in parallel.

This doesn't affect the learned parameters at all; just the activations. Basically, we just lift them into a tensor of 1 higher rank. Instead of representing the input as an $n \times d$ matrix, we'll represent it as a $b \times n \times d$ tensor.

The rest of the math is exactly the same. At the hardware level, this will just result in the same operations (including the same weights) being applied to different input at the same time. TPUs are highly optimized for this.
