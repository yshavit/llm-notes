# Algebraic reformulations

:::{status} 1
:::

I mentioned way back [in the introduction](#conceptual-layers) that I find it useful to think about LLMs first in terms of the fundamental concepts, and then in terms of the algebraic reformulations of those concepts. Until now, I've been focusing exclusively on the conceptual layers. In this chapter, I'll describe how those get bundled into mathematical objects that are more efficient to compute.

There are two major parts to this:

- Turning vectors-of-vectors into matrices
- Increasing the rank of all tensors by 1, so that we can add a batching dimension

:::{note} Assume $d = \delta$
In this chapter, I'll assume that each transformation's input and output dimensions are the same. I mentioned that this is a common model constraint in the chapter on attention, and the previous chapter provided additional motivation for it in the context of residual connections. Since it also simplifies this chapter's math, I'll just assume it from here on out.
:::

:::{warning} Warning! Math!
Parts of this chapter may be a bit dense.

Until now, we've been able to mostly get by with just understanding the shapes of various operations, like dot products and matrix multiplication. In this chapter, you'll need to understand the actual operations.

If you don't remember how these work, you may want to review the earlier chapter on [vector and matrix math](#matrix-math-details).
:::

## The architecture's conceptual shape

Before we dive into the algebraic reformulations, let's take a look at the LLM's architecture once more, this time focusing on the shapes of the learned parameters and activations. I'll skip the tokenization phase, since that's effectively a preparation step that happens before the LLM itself runs.

For most of the LLM, the activations are in the form of $n$ vectors, each size $d$. The final output is still $n$ vectors, but each sized $v$ (the vocabulary size).

:::{drawio} images/tensors/architecture-concepts
:alt: An overview of the LLM architecture, showing n vectors of size d for most of the flow, and a final output of n vectors of size v
:class: wide-image
:::

## Vectors of vectors → matrices

The basic "lifting" we'll do is to to turn vectors of vectors into matrices. This will let us turn the various "for each outer vector, do some stuff" loops that we've been working with into matrix multiplication (I'll describe each of these in detail below). This doesn't change what's going on conceptually, but it lets us do the math on GPUs that process it much more quickly.

All we need to do is turn each "outer" vector into a row in a matrix:

$$
\underbrace{
\begin{array}{llll}
[\; 1.32 \,, & 5.91 \,, & 5.71 \,, & \dots \;] \\[0.15em]
[\; 6.16 \,, & 4.81 \,, & 3.62 \,, & \dots \;] \\[0.15em]
[\; 8.27 \,, & 9.53 \,, & 2.44 \,, & \dots \;] \\[0.15em]
[\; \dots\,, & \dots\,, & \dots\,, & \dots \;] \\[0.50em]
\end{array}
}_{\vphantom{\big|}n \text{ vectors of size } d}
\quad \Longrightarrow \quad
\underbrace{
\begin{bmatrix}
1.32 & 5.91 & 5.71 & \dots \\
6.16 & 4.81 & 3.62 & \dots \\
8.27 & 9.53 & 2.44 & \dots \\
\vdots & \vdots & \vdots & \ddots
\end{bmatrix}
\rule[-2.75em]{0pt}{0pt}
}_{\vphantom{M}n \times d \text{ matrix}}
$$

Let's work through what that means for the calculations I've described in the previous chapters.

### Calculating attention

Recall that we calculated attention by doing a nested loop over the input embeddings:

1. For each input embedding $t_q$ (there are $n$ of them):
    1. Calculate the query vector $q = t_q \times W_q$. This vector has size $d$.
    2. For each input embedding $t_k$ (the same $n$ embeddings as for the query vector), calculate the attention score of $q$ against $t_k$:
        1. Calculate the key vector $k = t_k \times W_k$. This vector has size $d$.
        2. Calculate the dot product $q \cdot k$ to get the attention score (a scalar).
    3. Treat those $n$ attention scores as a vector; scale and softmax that vector to get the attention weight vector (size $n$).
    (calculate-value-vectors)=
    4. Calculate value vectors:
        1. For every input embedding $t_v$, calculate a value vector $v = t_v \times W_v$. There are $n$ such vectors, each size $d$.
        2. Multiply each value vector by the corresponding attention weight (the $n$ scalars from the previous step). The result is still $n$ vectors, each size $d$.
    5. Sum the value vectors to get the context vector. This vector has size $d$.

There are $n$ inputs (that is, $n$ iterations of the $t_q$ loop), so we ended up with $n$ context vectors, each of size $d$.

Let's see how much of this we can turn into matrix math. (Spoiler alert: almost all of it.) Instead of a nested loop that generates $n$ vectors of size $d$, we'll use matrix math to generate an $n \times d$ matrix.

#### Calculating the query matrix

I'll start with step 1.1 above. We'll focus on one iteration of the loop --- call it $i$ --- and calculate the key vector $q_i$ for input $t_i$. (Remember that $t_i$ is a $d$-sized embedding vector.)

$$
\begin{align}
q_i & = t_i \times W_k \\
    & = \underbrace{\begin{bmatrix} t_{i,1} & t_{i,2} & \dots \end{bmatrix}}_{1 \times d} \cdot \underbrace{W_k}_{d \times d} \\
    & = \begin{bmatrix} t_{i,1} & t_{i,2} & \dots \end{bmatrix} \cdot \begin{bmatrix} w_{1,1} & w_{1,2} & \dots \\ w_{2,1} & w_{2,2} & \dots \\ \vdots & \vdots & \ddots \end{bmatrix} \\
    & = \begin{bmatrix}
          \left(
            \begin{bmatrix} t_{i,1} & t_{i,2} & \dots \end{bmatrix}
            \begin{bmatrix} w_{1,1} \\ w_{2,1} \\ \vdots \end{bmatrix}
          \right) \;
        &
          \left(
          \begin{bmatrix} t_{i,1} & t_{i,2} & \dots \end{bmatrix}
          \begin{bmatrix} w_{1,2} \\ w_{2,2} \\ \vdots \end{bmatrix}
          \right) \;
        & \dots
      \end{bmatrix}
\end{align}
$$

If we do this for each embedding, we get a matrix that we'll call $Q$. This matrix represents the 1.1 step executed across all of the top-level iterations:

$$
\left.
\begin{array}{l}
\text{1. For each input embedding } t_q\\
\quad \text{1. Calculate the query vector } q = t_q \times W_q
\end{array}
\right\} Q_{(n \times d)}
$$

Let's put each of those iterations into a row of a matrix:

$$
\begin{align}
Q & = \left. \begin{bmatrix}
        t_1 \times W_q \\
        t_2 \times W_q \\
        \vdots
      \end{bmatrix} \right\} n \text{ rows} \\[2.5em]
  & = \underbrace{
        \begin{bmatrix}
          \left(
            \begin{bmatrix} t_{1,1} & t_{1,2} & \dots \end{bmatrix}
            \begin{bmatrix} w_{1,1} \\ w_{2,1} \\ \dots \end{bmatrix}
          \right)
        &
          \left(
            \begin{bmatrix} t_{1,1} & t_{1,2} & \dots \end{bmatrix}
            \begin{bmatrix} w_{1,2} \\ w_{2,2} \\ \dots \end{bmatrix}
          \right)
        & \dots
        \\[2.5em]
          \left(
            \begin{bmatrix} t_{2,1} & t_{2,2} & \dots \end{bmatrix}
            \begin{bmatrix} w_{1,1} \\ w_{2,1} \\ \dots \end{bmatrix}
          \right)
        &
          \left(
            \begin{bmatrix} t_{2,1} & t_{2,2} & \dots \end{bmatrix}
            \begin{bmatrix} w_{1,2} \\ w_{2,2} \\ \dots \end{bmatrix}
          \right)
        & \dots
        \\ \vdots & \vdots &\ddots
        \end{bmatrix}
        \rule[-5.25em]{0pt}{0pt}
      }_{d \text{ elements} }
\end{align}
$$

This looks like matrix multiplication --- and it is! Specifically, the $t_{i,j}$ elements make up a $X$ matrix whose rows are the $n$ inputs and whose columns are each input's $d$ embedding dimensions; and the $w_{k,l}$ elements represent the $d \times d$ weight matrix.

This means we can calculate $Q$ with just one matrix multiplication:

$$
Q = \underbrace{XW_q}_{n \times d} \\[1.5em]
\scriptstyle\textit{where $X$ is the input embedding}
$$

This is really powerful! It means the first part of the nested loop (steps 1 → 1.1) can be reduced to a single matrix multiplication, which GPUs are extremely efficient at processing. We'll be doing similar things for the key and value vectors, so I'd suggest taking the time to work through the above and make sure it makes sense to you.

#### Calculating attention scores matrix

Now, we can move onto the raw attention scores. This corresponds to step 1.2 above.

First, let's calculate the key matrix $K$. This is exactly the same as the query matrix $Q$, except that it uses $W_k$ instead of $W_q$. Because the progression from vectors-of-vectors to matrix is the same, I won't spell it out in full.

$$
K = \underbrace{XW_k}_{n \times d}
$$

Next, we'll calculate all the attention scores as a matrix. Each row will correspond to a query token, and each column will be the attention score between that query token and the corresponding key token:

$$
\begin{align}
\text{attention scores} & = \begin{bmatrix}
    q_1 \cdot k_1 & q_1 \cdot k_2 & \dots \\
    q_2 \cdot k_1 & q_2 \cdot k_2 & \dots \\
    \vdots & \vdots & \ddots
  \end{bmatrix} \\
& = \begin{bmatrix}
  Q_1 \cdot \text{(key vector 1)} & Q_1 \cdot \text{(key vector 2)} & \dots \\
  Q_2 \cdot \text{(key vector 1)} & Q_2 \cdot \text{(key vector 2)} & \dots \\
  \vdots & \vdots & \ddots
\end{bmatrix}
\end{align}
$$

Once again, this looks like matrix multiplication! The one problem is the key vectors. In that matrix multiplication for the attention scores, each $\textit{(key vector } i \textit{)}$ needs to be a $d$-sized vector corresponding to a horizontal row within $K$. But, if we calculated this matrix as $\textbf{attention scores} = QK$, then the thing that should be $d$-sized key vectors would instead be the $n$-sized vertical slices of $K$:

$$
= \begin{bmatrix}
  \left( Q_1 \begin{bmatrix}K_{1,1} \\ K_{2,1} \\ \vdots \end{bmatrix} \right) & \left( Q_1 \begin{bmatrix}K_{1,2} \\ K_{2,2} \\ \vdots \end{bmatrix} \right) & \dots \\[2.5em]
  \left( Q_2 \begin{bmatrix}K_{1,1} \\ K_{2,1} \\ \vdots \end{bmatrix} \right) & \left( Q_2 \begin{bmatrix}K_{1,2} \\ K_{2,2} \\ \vdots \end{bmatrix} \right) & \dots \\
  \vdots & \vdots & \ddots
\end{bmatrix}
$$

Not only is this not what we want, but the math isn't even defined: we're taking the dot products of $d$-sized $Q_i$ vectors and $n$-sized $K_{\star \,,\,j}$ vectors.

What we need is to replace the vertical slicing of $K$ with horizontal slicing. To do that, we just need to [transpose](#matrix-transposition) $K$. This turns its rows into columns --- meaning that when take vertical slices of the transposed $K^T$ matrix during multiplication, what we actually get are the rows of $K$.

Now we can just multiply $Q$ by $K^T$. For example, the first cell in this matrix would be:

$$
\begin{align}
\text{attention scores}_{(1,1)} & = Q_1 \begin{bmatrix}{K^T}_{1,1} \\ {K^T}_{2,1} \\ \vdots \end{bmatrix} \\[2.5em]
& = Q_1 \begin{bmatrix}K_{1,1} \\ K_{1,2} \\ \vdots \end{bmatrix}
\end{align}
$$

Now the math works out: we're multiplying $Q_{n \times d}$ by ${K^T}_{d \times n}$ to get an $n \times n$ matrix, the raw attention scores:

$$
\text{attention scores} = QK^T
$$

Just to belabor the point: we've turned all of the nested looping in steps 1 → (1.1 - 1.2) into just a few matrix operations:

1. $Q = XW_q$
2. $K = XW_k$
3. Transpose $K$ (this doesn't even require moving any memory: it's just a bit of metadata to tell the computer to treat $i,j$ as $j,i$)
4. $\text{attention scores} = QK^T$

(scale-and-softmax-matrix)=

#### Causal attention, scale, and softmax

Next, we just need to apply the causal mask, scale each element in the attention scores by dividing it by $\sqrt{d}$, and then apply softmax. This corresponds to step 1.3 above.

$$
\text{attention weights} = A = \text{softmax}\left( \frac{QK^T}{\sqrt{d}} \right)
$$

Note:

- To apply softmax, you can create an $n \times n$ matrix with 0s in the bottom-left and $-\infty$ in the top right:

  $$
  \begin{bmatrix}
  0 & -\infty & -\infty \\
  0 & 0 & -\infty \\
  0 & 0 & 0
  \end{bmatrix}
  $$

  Then, just add this to the attention scores matrix.
- Dividing a matrix by a scalar ($\sqrt{d}$) just divides each of its cells by that scalar.
- Softmax operates on vectors. When we apply it to a matrix, this really just means to applying it to each row in that vector. Each of those rows will have softmax calculated independently, but GPUs can parallelize the work efficiently across those rows.

None of these operations change the dimensions of the matrix, so it's still $n \times n$.

#### Context matrix

Finally, we'll apply our weights against the value vectors, and sum the results. This corresponds to step 1 → (1.4, 1.5) above.

First, we'll get the value matrix $V$, similar to the above. This is step 1.4.1.

$$
V = \underbrace{XW_v}_{n \times d}
$$

Each row in this matrix is one value vector.

Before we go further, let's step back and compute just a single context vector (that is, just a single token's attentions) the matrices we've computed so far.

Just to recap, here's what we need to do:

::::{blockquote}

1. For each input $t_q$:

   ...

   :::{embed} #calculate-value-vectors
   :::
::::

This means that within the context of a single query token $Q_i$, we need to:

- Take all the value vectors (step 1.4.1):

  $$
  \left.\begin{bmatrix} V_1 \\ V_2 \\ \vdots \end{bmatrix}\right\} \text{$n$ vectors, each size $d$}
  $$

  (Remember that $V$ is an $n \times d$ matrix; each row $V_i$ is a $d$-vector.)

- Multiply each value vector by its corresponding attention weight for query token $i$ (step 1.4.2):

  $$
  \begin{align}
  & \begin{bmatrix}A_{i,1} \, V_1 \\ A_{i,2} \, V_2 \\ \vdots \end{bmatrix} \\[2em]
  = & \left.
    \begin{bmatrix}
      A_{i,1} \, V_{1,1} & A_{i,1} \, V_{1,2} & \cdots \\
      A_{i,2} \, V_{2,1} & A_{i,2} \, V_{2,2} & \cdots \\
      \vdots & \vdots & \ddots
    \end{bmatrix}
    \right\} n
  \end{align}
  $$

- Sum the $n$ weighted vectors to get the context vector for query $i$ (step 1.5):

  $$
  \begin{bmatrix}
  \underbrace{A_{i,1} \, V_{1,1} + A_{i,2} \, V_{2,1} + \cdots}_{\text{weighted sum of column 1}}
  & \underbrace{A_{i,1} \, V_{1,2} + A_{i,2} \, V_{2,2} + \cdots}_{\text{weighted sum of column 2}}
  & \cdots
  \end{bmatrix}
  $$

We'll call this vector $C_i$, the context vector for the $i$-th query token. Let's see what all the $C_i$s look like stacked as rows of a matrix:

$$
\begin{align}
\begin{bmatrix} C_1 \\ C_2 \\ \vdots \end{bmatrix}
& = \begin{bmatrix}
(A_{1,1}V_{1,1} + A_{1,2}V_{2,1} + \cdots) & (A_{1,1}V_{1,2} + A_{1,2}V_{2,2} + \cdots) & \cdots \\
(A_{2,1}V_{1,1} + A_{2,2}V_{2,1} + \cdots) & (A_{2,1}V_{1,2} + A_{2,2}V_{2,2} + \cdots) & \cdots \\
\vdots & \vdots & \ddots
\end{bmatrix}
\end{align}
$$

Each cell $i,j$ is a sum of terms: each of row $A_i$'s columns multiplied by column $V_{\star,j}$'s rows. In other words, each cell is a dot product:

$$
= \begin{bmatrix}
  \begin{bmatrix} A_{1,1} & A_{1,2} & \cdots \end{bmatrix}
    \begin{bmatrix} V_{1,1} \\ V_{2,1} \\ \vdots \end{bmatrix}
  & \begin{bmatrix} A_{1,1} & A_{1,2} & \cdots \end{bmatrix}
    \begin{bmatrix} V_{1,2} \\ V_{2,2} \\ \vdots \end{bmatrix}
  & \cdots \\[2.5em]
    \begin{bmatrix} A_{2,1} & A_{2,2} & \cdots \end{bmatrix}
  \begin{bmatrix} V_{1,1} \\ V_{2,1} \\ \vdots \end{bmatrix}
  & \begin{bmatrix} A_{2,1} & A_{2,2} & \cdots \end{bmatrix}
    \begin{bmatrix} V_{1,2} \\ V_{2,2} \\ \vdots \end{bmatrix}
  & \cdots \\[2.5em]
  \vdots & \vdots & \ddots
\end{bmatrix}
$$

This may look familiar: it's just the matrix multiplication $AV$.

#### The full attention calculation

So, attention is $AV$. If we substitute $A$ with the expression from @scale-and-softmax-matrix above, we get:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d}} \right)V
$$

This is the canonical representation of attention, and is somewhat famous within the literature of LLMs.

This means we've now turned the all of the attention calculation --- a logically multiple-nested loop --- into a few matrix multiplications and a bit of parallelizable manipulation:

1. $Q = XW_q + b_q$
2. $K = XW_k + b_k$
3. $V = XW_v + b_v$
4. $\text{attention scores} = QK^T$
5. divide these by $\sqrt{d}$
6. apply softmax to each row to get $A$, the attention weight matrix
7. $\text{Attention} = AV$

A GPU is going to eat this for breakfast!

#### Multi-head attention

Back in the chapter on attention, I talked about how [LLMs use multiple heads](#multi-head) within a single attention layer, each learning a different relationship. The attention layer concatenates these heads, and then uses a final projection $W_o$ to combine them.

Described as such, this would require looping over each of the heads to perform the attention function we just saw. It may not surprise you that this can be done without looping, using tensor math.

:::{note} Notation
For this section, within matrices I'll be using Latin letters (i.e., "normal" letters) for to represent weight parameters, and Greek letters to represent activations. I'll also use Latin letters for dimensions, as I've been doing throughout this chapter.
:::

First, let's refresh the multi-head ideas:

- $n$ is the input length, in tokens
- overall dimensionality is still $d$ (for both input and output)
- $h$ heads
- each head is $d \times \frac{d}{h}$

To illustrate everything, I'll pick $n = 3$, $d = 4$, and $h = 2$.

First, for each of $W_q$, $W_k$, and $W_v$, we'll concatenate the heads' weights to create a single, $d \times d$ matrix. For example:

$$
W_q =
\underbrace{
  \begin{bmatrix}
  a & b \\
  e & f \\
  i & j \\
  m & n
  \end{bmatrix}
}_{\text{head 1}}
\underbrace{
  \begin{bmatrix}
  c & d \\
  g & h \\
  k & l \\
  o & p
  \end{bmatrix}
}_{\text{head 2}}
\rightarrow
\begin{bmatrix}
a & b & c & d \\
e & f & g & h \\
i & j & k & l
\end{bmatrix}
$$

(Remember, each head is $d \times \frac{d}{h}$ --- so in our example, $4 \times \frac{4}{2}$, a.k.a $4 \times 2$.) Note that this doesn't happen at runtime, during inference: these are learned parameters, so we can lay them out this way as we build the model.

At inference, we'll multiply the input by these matrices, just as we did in the single-head description above. So for example:

$$
Q = \underbrace{XW_q}_{(n \times d)\,(d \times d)}
= \underbrace{
  \begin{bmatrix}
  \alpha   & \beta  & \gamma  & \delta \\
  \epsilon & \zeta  & \eta    &\theta \\
  \iota    & \kappa & \lambda & \mu
  \end{bmatrix}
}_{n \times d}
$$

Remember that in matrix multiplication, the each cell in the result combines the corresponding row from the left matrix (the input, for us) and the column from the right matrix (the weights). Since our weights were split up column-wise by heads, the corresponding matrix products are, too:

(split-q-by-head)=

$$
Q =
\left[\begin{array}{cc|cc}
  \alpha   & \beta  & \gamma  & \delta \\
  \epsilon & \zeta  & \eta    &\theta \\
  \iota    & \kappa & \lambda & \mu
\end{array}\right]
$$

At this point, we need to do actual looping --- not just clever matrix math. For each of the heads, we'll compute attention just as we did above:

$$
\text{Head Attention}(Q_h, K_h, V_h) = \text{softmax}\left( \frac{Q_h{K_h}^T}{\sqrt{d/h}} \right)V_h
$$

(Remember to [divide the scaling factor by $h$](#multi-head-scaling) to account for each head's smaller embedding dimension!) Let's look at the shape of this head attention. We can disregard softmax and $\sqrt{d/h}$ (they don't change the shape of vectors or matrices), in which case we get:

$$
\begin{align}
\text{Head Attention}(Q_h, K_h, V_h) & = \sout{\text{softmax}}\left( \frac{Q_h{K_h}^T}{\sout{\sqrt{d/h}}} \right)V_h \\
& = ( Q_h{K_h}^T )V_h \\
& = \left[ \left(n \times \frac{d}{h}\right) \left(n \times \frac{d}{h} \right)^T \right] \left(n \times \frac{d}{h} \right) \\
& = \left[ \left(n \times \frac{d}{h} \right) \left(\frac{d}{h} \times n \right) \right] \left(n \times \frac{d}{h} \right) \\
& = \left(n \times n \right) \left(n \times \frac{d}{h} \right) \\
& = n \times \frac{d}{h}
\end{align}
$$

So with all that, we now have $h$ head attentions, each sized $n \times \frac{n}{h}$. Now we reverse the process that we took with the weights: we take these head attentions, concatenate them by columns, and treat them as a single attention output:

$$
\text{Attention} =
\left[\begin{array}{cc|cc}
\nu & \xi & \omicron & \pi \\
\rho & \sigma & \tau & \upsilon \\
\phi & \chi & \psi & \omega \\
\end{array}\right]
$$

In this figure, each "side" of the attention represents the output from one head, sized $n \times \frac{d}{h}$. The concatenated heads form a single, $n \times d$ matrix.

If you recall, the last step in the multi-head process was to [multiply the output by a $W_o$ matrix](#w-o-projection). This is just a $d \times d$ matrix, so there's nothing special to do here: we just apply the matrix multiplication.

#### Combined QKV matrix

In the above (and back in our original chapter on attention), we treated the $W_q$, $W_k$, and $W_v$ weight matrices as three separate matrices. To calculate the query, key, and value matrices, we did:

1. $Q = XW_q + b_q$
2. $K = XW_k + b_k$
3. $V = XW_v + b_v$

In practice, these are usually concatenated into one matrix, $W_{qkv}$:

$$
W_{qkv} = \begin{bmatrix}
q & q & q & \cdots & k & k & k & \cdots & v & v & v & \\
q & q & q & \cdots & k & k & k & \cdots & v & v & v & \\
q & q & q & \cdots & k & k & k & \cdots & v & v & v & \\
\end{bmatrix}
$$

(Note that for brevity, I'm being a bit informal in my notation here: in particular, I'm writing the various $q_{i,j}$ values as just $q$, and similarly for $k$ and $v$). We apply matrix multiplication and addition to this:

$$
XW_{qkv} = \begin{bmatrix}
T_1q + b_q & \cdots & T_1k + b_k & \cdots T_1v + b_v \\
T_2q + b_q & \cdots & T_2k + b_k & \cdots T_2v + b_v \\
T_3q + b_q & \cdots & T_3k + b_k & \cdots T_3v + b_v \\
\end{bmatrix}
$$

... and then just split the matrix into three slices:

$$
W, K, V =
\begin{bmatrix}
T_1q + b_q & \cdots \\
T_2q + b_q & \cdots \\
T_3q + b_q & \cdots \\
\end{bmatrix}
,
\begin{bmatrix}
T_1k + b_k & \cdots \\
T_2k + b_k & \cdots \\
T_3k + b_k & \cdots \\
\end{bmatrix}
,
\begin{bmatrix}
T_1v + b_v & \cdots \\
T_2v + b_v & \cdots \\
T_3v + b_v & \cdots \\
\end{bmatrix}
$$

This lets us to do all three matrix multiplications ($W$, $K$, and $V$) in a single operation. GPUs have some fixed overhead in any given matrix multiplication, so this optimization just amortizes that overhead across all three matrices.

#### KV caching

In all of the above, we've been calculating the full $n \times d$ attention for an input $n$ tokens. When we process the user's initial prompt, this is great. But as we generate tokens, we can calculate attention incrementally.

For example, let's say the user entered {keyboard}`The` {keyboard}`quick` {keyboard}`brown` {keyboard}`fox`. This prompt is 4 tokens, so our attention is $4 \times d$. We generate the next token, {keyboard}`jumps`, and then loop back for another round of inference. The naive, full-attention calculation I've been describing so far would require a $5 \times d$ attention, for {keyboard}`The` {keyboard}`quick` {keyboard}`brown` {keyboard}`fox` {keyboard}`jumps`. We can short-circuit much of this calculation.

Let's take a quick review of everything we did above. To make things concrete, I'll pick $d = 2$, and we'll look at a 3-sequence input ($n = 3$).

First, we calculate $Q$, $K$, $V$. These are all $n \times d$ matrices (the products of the $n \times d$ input and the $d \times d$ weight matrices):

$$
% Q
\begin{bmatrix}
Q_{1,1} & Q_{1,2} \\
Q_{2,1} & Q_{2,2} \\
Q_{3,1} & Q_{3,2}
\end{bmatrix}
\quad
% K
\begin{bmatrix}
K_{1,1} & K_{1,2} \\
K_{2,1} & K_{2,2} \\
K_{3,1} & K_{3,2}
\end{bmatrix}
\quad
% V
\begin{bmatrix}
V_{1,1} & V_{1,2} \\
V_{2,1} & V_{2,2} \\
V_{3,1} & V_{3,2}
\end{bmatrix}
$$

Next, we'll calculate $A' = QK^T$, which an $n \times n$ matrix:

$$
\begin{align}
A' = QK^T
& =
  \begin{bmatrix}
    Q_{1,1} & Q_{1,2} \\
    Q_{2,1} & Q_{2,2} \\
    Q_{3,1} & Q_{3,2}
  \end{bmatrix}
  \begin{bmatrix}
    K_{1,1} & K_{2,1} & K_{3,1} \\
    K_{1,2} & K_{2,2} & K_{3,2}
  \end{bmatrix} \\[1.5em]
& =
\begin{bmatrix}
  Q_{1,\star} K_{{1,\star}}
    & Q_{1,\star} K_{{2,\star}}
    & Q_{1,\star} K_{{3,\star}} \\
  Q_{2,\star} K_{{1,\star}}
    & Q_{2,\star} K_{{2,\star}}
    & Q_{2,\star} K_{{3,\star}} \\
  Q_{3,\star} K_{{1,\star}}
    & Q_{3,\star} K_{{2,\star}}
    & Q_{3,\star} K_{{3,\star}}
\end{bmatrix}
\end{align}
$$

(Note that I'm using informal, nonstandard notation here: $M_{1,\star}$ to represent row 1 of $M$, and $M_{\star,1}$ to represent column 1. Also, for simplicity, I'm omitting causal attention, scaling, and softmax --- we don't need them right now. That means $A'$ isn't quite the $A$ we've been using above.)

Finally, we calculate $A'V$, which is $n \times d$:

$$
A'V = \begin{bmatrix}
A'_{1,\star}V_{\star,1} & A'_{1,\star}V_{\star,2} \\[0.5em]
A'_{2,\star}V_{\star,1} & A'_{2,\star}V_{\star,2} \\[0.5em]
A'_{3,\star}V_{\star,1} & A'_{3,\star}V_{\star,2}
\end{bmatrix}
$$

Remember that when our round of inference is done, we're only going to [use the last logit](#using-last-logit), which will be derived just from the last row of this attention (after passing it through various FFNs and other transformer blocks). So, let's focus on the last row of $AV$.

$$
(A'V)_3 =
\begin{bmatrix}
A_{3,\star}V_{\star,1} & A_{3,\star}V_{\star,2}
\end{bmatrix}
$$

As a reminder, $A_3$ is:

$$
\begin{bmatrix}
  Q_{3,\star} K_{{1,\star}}
    & Q_{3,\star} K_{{2,\star}}
    & Q_{3,\star} K_{{3,\star}}
\end{bmatrix}
$$

This means that $(A'V)_3$ contains:

- all of $K$'s data ($K_{n,\star}$ for every token $n$)
- all of $V$'s data ($V_{\star,d}$ for every dimension $d$)
- only the $Q_{3,\star}$ row

So far, this is just a reshash of everything we've already seen. Here's where it gets interesting! We can make some observations:

- Each row $n$ in $K$ is independently calculated, based on the $W_k$ weights and the $n$th token's embedding. In other words, each token stays within its row in $K$.
- This means we can cache $K$ at every round of inference. In the next round, we don't need to calculate all of $K$ from scratch: the cache gives us the first $n-1$ rows. All we need to calculate is the $n$th row.
- Similarly for $V$.
- For $Q$, we don't need the first $n-1$ rows at all: all we need is the $n$th row.

We do still need to build the full $K$ and $V$ matrices; we just don't need to compute most of them, since all but the last row are cached. For $Q$, we don't even need to build the full matrix.

Also, since we're now only computing the last row of attention, we don't need to account for causal attention. Remember that the attention mask [only zeroed out weights for rows before the last row](#causal-attention-grid); the last row is unaffected, and that's the only one we're generating.

Putting it all together, we have essentially the same attention formula as before, but tweaked to only generate the last row:

$$
\text{Attention}(Q_n, K, V) = \text{softmax}\left( \frac{Q_nK^T}{\sqrt{d/h}} \right)V
$$

- $Q_n$ is a $1 \times d$ matrix, derived from just the most recent token (a $1 \times d$ embedding) and the $W_q$ weights
- $K$ is constructed by taking the cached $K$ --- an $(n-1) \times d$ matrix -- and appending the $1 \times d$ matrix that's the most recent token multiplied by $W_k$.
- $V$ is similarly constructed

So:

- $Q_nK^T$ is $(1 \times d) (d \times n) = (1 \times n)$
- softmax and scaling maintain these dimensions
- $AV$ is $(1 \times n)(n \times d) = (1 \times d)$

And there we have it! We've calculated just the last row in attention, which will then snake through the FFN and other transformer blocks to produce a single logit, the next prediction.

:::{warning} Beware position offsets!
If you implement KV caching, each round of inference will only see one token. If you're not careful, this can break your [position embeddings](#position-embeddings), since every token will think it's at position 1!

Just make sure to keep a count of how many tokens you've seen, so that you can use that as the position embedding index.
:::

#### Implementation details

:::{note}
If you're not interested in how to translate this to actual code, you can skip this section and move straight to the [discussion of FFNs below](#algebraic-reformulations-ffn).
:::

To do all of the matrix concatenations efficiently, we need to get into the nitty-gritty of standard matrix libraries in software. This book doesn't cover any particular library, but they'll all work pretty similarly.

We'll pick up where we split the $Q$ matrix by head:

:::{embed} #split-q-by-head
:::

Tensor libraries will let you reinterpret one tensor as another, differently-shaped one. In our case, we're going to reinterpret the $n \times d$ matrix (a rank 2 tensor) into an $n \times h \times \frac{d}{h}$ tensor (rank 3), which splits it up just as we've just visualized.

:::{seealso} Why does the reinterpretation work like that?
:class: dropdown

Internally, tensor libraries typically store the matrix as a single, contiguous array of values:

$$
\alpha \ \delta \ \eta \ \kappa \ \beta \ \epsilon \ \theta \ \lambda \ \gamma \ \zeta \ \iota \ \mu
$$

The first dimension splits this evenly among the dimension size (in this case, $n = 3$, so three groups):

$$
\alpha \ \delta \ \eta \ \kappa
\qquad \beta \ \epsilon \ \theta \ \lambda
\qquad \gamma \ \zeta \ \iota \ \mu
$$

The next dimension splits each of these groups, again evenly depending on size (in this case, $h = 2$, so two sub-groups):

$$
\alpha \ \delta \quad \eta \ \kappa
\qquad \beta \ \epsilon \quad \theta \ \lambda
\qquad \gamma \ \zeta \quad \iota \ \mu
$$

If we had even higher dimensions (4-rank tensors and above), we would keep going.

The last dimension is always just "the rest of the elements at this level" --- or more colloquially, we can think of it as the column dimension.

This is all a bit of a simplification: things like transposition and optimization details can complicate the picture. But at a high level, it's a good way of understanding what's going on.
:::

When tensor libraries perform matrix operations on higher-order tensors (rank 3 or above), they treat the leftmost dimensions as "batching dimensions" --- basically, dimensions to loop over. They can load this batching to GPUs, where it happens very efficiently.

Unfortunately, this approach doesn't quite work for our $n \times h \times \frac{d}{h}$ tensor: we want to loop over each of the $h$ heads, not each of the $n$ rows. To solve this, we'll first transpose the tensor to $h \times n \times \frac{d}{h}$. This doesn't change its layout at all: it just changes how the library indexes into the tensor, and thus how it batches.

To summarize, we've done three things:

- multiplied $TQ$ to get an $n \times d$ matrix
- reinterpreted that as an $n \times h \times \frac{d}{h}$ tensor
- transposed that to a $h \times n \times \frac{d}{h}$ tensor

Now we just apply the attention formula:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d/h}} \right)V
$$

This time, $Q$, $K$, and $V$ are each those 3-rank tensors, with $h$ as the batch dimension. When they're multiplied together, the libraries will match each batch up. For example, to multiply $QK^T$:

- $Q$ is $h \times n \times \frac{d}{h}$
- $K^T$ is $h \times \frac{d}{h} \times n$ (you may have to do this transposition explicitly)
- When the library multiplies the two, it'll first multiply batch 0 from $Q$ with batch 0 from $K^T$ to produce batch 0 of the result. Then it multiplies $Q$ batch 1 with $K^T$ batch 1 to produce result batch 1, and so on.

When you apply the $\text{softmax}$ function, you'll explicitly tell the library which dimension to apply it against (in our case, the columns --- that is, the last dimension). $\sqrt{d/h}$ only applies to scalars, so it doesn't need any dimension or batch handling; it just applies independently to each of the values.

The result of all that is an attention tensor, which is $h \times n \times \frac{d}{h}$. Now we just reverse the reshaping: we transpose this to $n \times h \times \frac{d}{h}$ and then reinterpret it as a rank-2, $n \times d$ matrix.

(algebraic-reformulations-ffn)=

### FFNs

As I mentioned [in the previous chapter](#typical-ffn), each FFN in an LLM typically consists of the input sized $d$, one hidden layer sized $4d$, and an output layer sized $d$. The FFN's input and output correspond to a single token embedding; this gets evaluated [separately for each token](#ffn-output-shape), though GPUs are able to do those separate evaluations efficiently in parallel.

Let's look at the FFN from the perspective of one layer. Remember from [the chapter on FFNs](#ffn-overview-diagram) that each layer has:

- an input vector of scalars, sized $d_{in}$
- $d_{out}$ neurons, each containing a $d_{in}$-sized vector of weights
- for each neuron, we:
  - calculate the dot product of the input and that neuron's weights; this gives us a scalar
  - add a scalar bias, one per neuron
  - pass that through an activation function to get one scalar per neuron, which is that neuron's activation

We can visualize the neuron weights as $d_{out}$ column vectors, each with $d_{in}$ elements:

$$
d_{in} \text{ weights}
\left\{
  \vphantom{\begin{matrix} \\ \\ \\ \\ \end{matrix}}
\right.
\underbrace{
  \begin{bmatrix} \alpha \\ \beta \\ \gamma \\ \delta \end{bmatrix}
  \begin{bmatrix} \epsilon \\ \zeta \\ \eta \\ \theta \end{bmatrix}
  \begin{bmatrix} \iota \\ \kappa \\ \lambda \\ \mu \end{bmatrix}
}_{
  \substack{\text{$d_{out}$ sets of weights,} \\[.5em] \text{one per neuron}}
}
$$

You may already see where this is going: we can treat this as a single $d_{in} \times d_{out}$ matrix. I'll call this matrix $W$.

We can also treat the layer's $d_{in}$-vector as a $1 \times d_{in}$ matrix, which I'll call X. If we do, we see that the matrix multiplication $XW$ gives us the right shape:

$$
\underbrace{X}_{1 \times d_{in}}
\cdot
\underbrace{W}_{d_{in} \times d_{out}}
= \underbrace{\text{layer}
}_{\substack{1 \times d_{out} \text{ matrix} \\[.5em] \Downarrow \\ d_{out}\text{ vector} }}
$$

Furthermore, each column in the output is the right value for the pre-bias neuron activation. For every column $j$ in $XW$, its value is:

$$
\begin{bmatrix}X_{1,\,1} & X_{1,\,2} & \dots & X_{1,\,d_{in}} \end{bmatrix}
\begin{bmatrix}W_{1,\,j} \\ W_{2,\,j} \\ \vdots \\ W_{d_{in},\,j} \end{bmatrix}
$$

Now we need to add the biases. There are $d_{out}$ of them, one per neuron. Instead of treating them as separate values and adding them one at a time,we'll treat them as a single $1 \times d_{out}$ matrix, and [add this](#adding-matrices) to the $1 \times d_{out}$ result from $jW$. I'll call this bias matrix $B$.

After that, we just need to apply the activation function. This does have to be applied to each value separately, but GPUs can efficiently parallelize that work.

This gives the full representation of each FFN layer:

$$
\text{Layer} = \text{Activation}( XW + B )
$$

To create the full FFN, we just apply each layer serially.

One crucial optimization we can make is to do all of the tokens' $XB$ calculation at once. Remember that $X$ is a $1 \times d_{in}$ matrix, corresponding to a single token in the prompt. If we consider the whole prompt, this is an $n \times d_{in}$ matrix:

$$
\begin{bmatrix}
X_{1,\,1} & X_{1,\,2} & \dots  & X_{1,\,d_{in}} \\
X_{2,\,1} & X_{2,\,2} & \dots  & X_{2,\,d_{in}} \\
\vdots   & \vdots   & \ddots & \vdots \\
X_{n,\,1} & X_{n,\,2} & \dots  & X_{n,\,d_{in}} \\
\end{bmatrix}
$$

If we multiply this by the $d_{in} \times d_{out}$ matrix $W$, the result will have $n$ rows, each corresponding to one row from the input $X$, and representing that row multiplied by the weight matrix $W$.

We still need to conceptually loop over each of those rows to add $B$, and then over every value to apply the activation function. GPUs can handle both of those efficiently, though.

### Normalization

Recall that for each embedding token, the normalization layer is calculated as:

:::{embed} #normalization-function
:::

We need to apply this to each input embedding separately. Unfortunately, here there's nothing tricky we can do with matrix math: not only does each embedding need to be evaluated separately, but calculating the mean and variance requires per-element calculations.

Luckily, the calculations themselves are pretty simple. And, as before, GPUs can handle these efficiently and in parallel.

The $n$ tokens of embedding $d$ _do_ get treated as a single $n \times d$ matrix. This is partially because GPUs know how to parallelize work efficiently across rows of matrices. It's also convenient for feeding the normalization into the attention layer, which as we've seen does gain from seeing the whole input as a single $n \times d$ matrix.

## Batching

Up until now, we've been working with one input at a time. In practice, GPUs can process multiple inputs in parallel.

This doesn't affect the learned parameters at all; just the activations. Basically, we just lift them into a tensor of 1 higher rank. Instead of representing the input as an $n \times d$ matrix, we'll represent it as a $b \times n \times d$ tensor.

The rest of the math is exactly the same. At the hardware level, this will result in the same operations (including the same weights) being applied to different inputs at the same time. GPUs are highly optimized for this.

## The final architecture

Our LLM now has essentially the same architecture as before: the only real difference is that we're treating the inputs not as $n$ vectors of size $d$ vectors, but a single $n \times d$ matrix. Similarly, the output is an $n \times v$ matrix. This lets us reformulate the operations we've already seen as matrix operations instead of logical loops, which lets us compute them far more efficiently on GPUs.

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

If someone were to provide you good values for all the weights throughout the architecture (and a _lot_ of AWS credits {emoticon}`;-)`), you'd have enough to build an LLM that would have been competitive in early 2020. You're not about to take down OpenAI or Anthropic, but that's still pretty neat!
