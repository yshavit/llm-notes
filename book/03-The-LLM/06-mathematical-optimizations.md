# Mathematical optimizations

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
