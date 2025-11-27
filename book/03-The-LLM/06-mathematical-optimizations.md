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

## Input vectors

:::{warning} TODO
From input vectors
:::

Each input token's embedding is a vector, but our input is a list of input tokens. This means we have a vector of vectors. We can group these into matrices. For example:

$$
\begin{array}{llllll}
\text{to} &  [ & 1.32, & 5.91, & 5.71, & \dots] \\
\text{be} &  [ & 6.16, & 4.81, & 3.62, & \dots] \\
\text{or} &  [ & 8.27, & 9.53, & 2.44, & \dots] \\
\text{...} & [ & \dots, & \dots, & \dots, & \dots]
\end{array}
\quad \Longrightarrow \quad
\begin{bmatrix}
1.32 & 5.91 & 5.71 & \dots \\
6.16 & 4.81 & 3.62 & \dots \\
8.27 & 9.53 & 2.44 & \dots \\
\dots & \dots & \dots & \dots
\end{bmatrix}
$$

Furthermore, during training we often want to work with batches of inputs at the same time. To do that, we just raise these matrices to rank 3 tensors, where the first index represents the input within the batch, and the other two represent the each input's matrix (which in turn just represents the embeddings for each token in the input). Again, we do this for the token embeddings, positional embeddings, and input embeddings.

## Attention

:::{important} TODO
brief discussion of implementation: the fact that the conceptual matrices get lifted into tensors with an extra dimension, for batching

Also, mention that attention weight is often given by the canonical formula:

$$
\text{Attention}(q, K, V) = \text{softmax}\left(\frac{qK^T}{\sqrt{d_k}}\right)V
$$

That formula is for a single query token's vector, as we've been working with. Since we'll eventually be doing this for all of the input tokens, treating each one as the query token in turn, we can represent that vector of query vectors $q$ as a matrix $Q$, giving us the even more canonical formula:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

Note that $X^T$ is the transpose of $X$. From claude:

> In matrix form:
>
> - $Q$ is an $n \times \delta$ matrix (one row per query token)
> - $K$ is an $n \times \delta$ matrix (one row per key token)
> - $K^T$ is $\delta \times n$ (transposed - rows become columns)
>
> So $QK^T$ gives you an $n \times n$ matrix - which is exactly your attention scores matrix! Each element $(i,j)$ is the dot product of query $i$ with key $j$.
>
> For the single-query version:
>
> - $q$ is a $1 \times \delta$ vector (or just $\delta$-dimensional)
> - $K$ is $n \times \delta$
> - $K^T$ is $\delta \times n$
> - $qK^T$ is $1 \times n$ (one attention score per token)
>
> The transpose is how you get the dot product of one query against all keys in one matrix operation, rather than computing them individually in a loop.

:::
