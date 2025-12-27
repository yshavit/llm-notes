# Algebraic reformulations

I mentioned way back [in the introduction](#conceptual-layers) that I find it useful to think about LLMs first in terms of the fundamental concepts, and then in terms of the algebraic reformulations of those concepts. Until now, I've been focusing exclusively on the conceptual layers. In this chapter, I'll describe how those get bundled into mathematical objects that are more efficient to compute.

There are two major components to this:

- Turning vectors-of-vectors into matrices
- Increasing the rank of all tensors by 1, so that we can add a batching dimension

:::{note} Assume $d = \delta$
In this chapter, I'll be relying on the common constraint that each transformation's input and output dimensions are the same. I mentioned that this is a common model constraint in the chapter on attention, and the previous chapter provided additional motivation for it in the context of residual connections. Since it also simplifies this chapter's math, I'll just assume it from here on out.
:::

## The architecture's conceptual shape

Before we dive into the algebraic reformulations, let's take a look at the LLM's architecture once more, this time focusing on the shapes of the learned parameters and activations.

I'll skip the tokenization phase, since that's effectively a preparation step that happens before the LLM itself runs.

:::{drawio} images/tensors/architecture-concepts
:alt: An overview of the LLM architecture, showing n vectors of size d for most of the flow, and a final output of n vectors of size v
:::

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

Back in the chapter on attention, I talked about how [LLMs use multiple heads](#multi-head) within a single attention layer, each (hopefully!) learning different patterns. The attention layer concatenates these heads, and then uses a final projection $W_o$ to combine them.

Described as such, this would require looping over each of the heads to perform the attention function I just described. It may not surprise you that this can be done without looping, using tensor math!

- Instead of the weights being $d_{model} \times d$, they're $d_{model} \times d_{model}$; in other words, each weight matrix contains the full, multi-head parameters.
- When we multiply the input $X$ against these, we get matrices of size $n \times d_{model}$
- We "reshape" these into rank-3 tensors $(n, h, d)$. This basically just means conceptually splitting along the columns:

  $$
  \begin{bmatrix}
  a & b & c & d \\
  e & f & g & h \\
  i & j & k & l
  \end{bmatrix}
  \rightarrow
  \underbrace{
    \begin{bmatrix}
    a & b \\
    e & f \\
    i & j
    \end{bmatrix}
  }_{\text{head 0}}
  \underbrace{
    \begin{bmatrix}
    c & d \\
    g & h \\
    k & l
    \end{bmatrix}
  }_{\text{head 1}}
  $$
- We then transpose those to $(h, n, d)$. This doesn't change the shape or contents of the heads, it just changes how we index them. At this point, each head is an $n \times d$ matrix.
- Now we calculate the attention weights $A$ as we did before.
  - The tensor libraries conceptually treat the first dimension ($h$, in our case) as a batching dimension; but the actual implementation is highly optimized.
  - The result is an $(h, n, n)$ tensor.
- We then multiply this by our $V_{(h,n,d)}$ to get an attention output $(h, n, d)$
- And finally, we transpose this back to $(n, h, d)$, reshape it back to $(n, d_{model})$ and apply the $W_o$ projection.

These operations are highly optimized in the software that runs them, and down to the hardware level.

### FFNs

Recall that [in the FFN](#ffn-overview-diagram), each layer has:

- an input vector of scalars, sized $d_{in}$
- $d_{out}$ neurons, each containing a $d_{in}$-sized vector of weights
- for each neuron, we:
  - calculate the dot product of the input and that neuron's weights; this gives us a scalar
  - add a scalar bias, one per neuron
  - pass that through an activations to get one scalar per neuron, which is that neuron's activation

Since this takes an input vector of scalars, this corresponds to a single embedding. As above, the full input is thus an $n \times d_{in}$ matrix. We can represent the neuron weights as a $d_{in} \times d_{out}$ matrix, which I'll call $W$ (this is not a standard term; there isn't really a standard term for these weights).

Since the first step of the FFN is to calculate the dot product of the input vector each column in $W$, we can calculate all of those dot products at once via the matrix multiplication $XW$. We can then add the biases as a $d_{out}$-sized vector $b$. Applying the activation to each of these gives us the full matrix-ified layer:

$$
\text{Layer} = \text{activation}( XW + b )
$$

The activation function is applied to each element in the matrix; but GPUs and TPUs can do this in parallel and very efficiently.

### Normalization

Recall that for each embedding token, normalization layer is calculated as:

:::{embed} #normalization-function
:::

To matrix-ify this, we'll just take our input matrix X ($n \times d$) and apply the normalization function per row. This still requires various per-element operations, but GPUs and TPUs can process each row in parallel, and the operations themselves are highly optimized.

## Batching

Up until now, we've been working with one input at a time. In practice, GPUs and especially TPUs can process multiple inputs in parallel.

This doesn't affect the learned parameters at all; just the activations. Basically, we just lift them into a tensor of 1 higher rank. Instead of representing the input as an $n \times d$ matrix, we'll represent it as a $b \times n \times d$ tensor.

The rest of the math is exactly the same. At the hardware level, this will just result in the same operations (including the same weights) being applied to different inputs at the same time. TPUs are highly optimized for this.

## The final architecture

Our LLM now has essentially the same architecture as before: the only real difference is that we're treating the inputs not as $n$ $d-sized$ vectors, but a single $n \times d$ matrix. Similarly, the output is an $n \times v$ matrix.

:::{drawio} images/tensors/architecture-matrix
:alt: The same architecture as above, but with matrices instead of vectors-of-vectors
:::

This diagram elides some of the complication, especially in the attention layer (and specifically, its multi-head architecture, as described above).

:::{aside}
:class: big

🎉

&nbsp;&nbsp;&nbsp;&nbsp;🎉

&nbsp;&nbsp;🎉
:::

That's it! **You have an LLM!**

If someone were to provide you good values for all the weights throughout the architecture, you'd have enough to build an LLM that would have been competitive in early 2020. You're not about to take down OpenAI or Anthropic, but that's still pretty neat!
