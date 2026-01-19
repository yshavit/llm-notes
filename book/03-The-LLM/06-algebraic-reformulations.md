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

The basic "lifting" we'll do is to to turn vectors of vectors into matrices. This will let us turn the various "for each outer vector, do some stuff" loops that we've been working with into matrix multiplication (I'll describe each of these in detail below). This doesn't change what's going on conceptually, but it lets us do the math on GPUs and TPUs that process it much more quickly.

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

This looks like matrix multiplication --- and it is! Specifically, the $t_{i,j}$ elements make up a $T$ matrix whose rows are the $n$ inputs and whose columns are each input's $d$ embedding dimensions; and the $w_{k,l}$ elements represent the $d \times d$ weight matrix.

This means we can calculate $Q$ with just one matrix multiplication:

$$
Q = \underbrace{TW_q}_{n \times d}
$$

This is really powerful! It means the first part of the nested loop (steps 1 → 1.1) can be reduced to a single matrix multiplication, which GPUs and TPUs are extremely efficient at processing. We'll be doing similar things for the key and value vectors, so I'd suggest taking the time to work through the above and make sure it makes sense to you.

#### Calculating attention scores matrix

Now, we can move onto the raw attention scores. This corresponds to step 1.2 above.

First, let's calculate the key matrix $K$. This is exactly the same as the query matrix $Q$, except that it uses $W_k$ instead of $W_q$. Because the progression from vectors-of-vectors to matrix is the same, I won't spell it out in full.

$$
K = \underbrace{TW_k}_{n \times d}
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

1. $Q = TW_q$
2. $K = TW_k$
3. Transpose $K$ (this doesn't even require moving any memory: it's just a bit of metadata to tell the computer to treat $i,j$ as $j,i$)
4. $\text{attention scores} = QK^T$

(scale-and-softmax-matrix)=

#### Scale and softmax

Next, we just need to scale each element in the attention scores by dividing it by $\sqrt{d}$, and then apply softmax. This corresponds to step 1.3 above.

$$
\text{attention weights} = A = \text{softmax}\left( \frac{QK^T}{\sqrt{d}} \right)
$$

Note:

- Dividing a matrix by a scalar ($\sqrt{d}$) just divides each of its cells by that scalar.
- Softmax operates on vectors. When we apply it to a matrix, this really just means to applying it to each row in that vector. Each of those rows will have softmax calculated independently, but GPUs and TPUs can parallelize the work efficiently across those rows.

Neither the scalar division nor softmax changes the dimensions of the matrix, so it's still $n \times n$.

#### Context matrix

Finally, we'll apply our weights against the value vectors, and sum the results. This corresponds to step 1 → (1.4, 1.5) above.

First, we'll get the value matrix $V$, similar to the above. This is step 1.4.1.

$$
V = \underbrace{TW_v}_{n \times d}
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

1. $Q = TW_q$
2. $K = TW_k$
3. $V = TW_v$
4. $\text{attention scores} = QK^T$
5. divide these by $\sqrt{d}$
6. apply softmax to each row to get $A$, the attention weight matrix
7. $\text{Attention} = AV$

A TPU is going to eat this for breakfast!

#### Multi-head attention

Back in the chapter on attention, I talked about how [LLMs use multiple heads](#multi-head) within a single attention layer, each (hopefully!) learning a different relationship. The attention layer concatenates these heads, and then uses a final projection $W_o$ to combine them.

Described as such, this would require looping over each of the heads to perform the attention function we just saw. It may not surprise you that this can be done without looping, using tensor math.

- Instead of the weights being $d \times d$, they're $d \times dh$; in other words, each weight matrix contains the full, multi-head parameters.
- When we multiply the input $T$ (an $n \times d$ matrix) against the multi-head weights, we get matrices of size $n \times dh$. These are the new, multi-head $Q$, $K$, and $V$ matrices.
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
  }_{\text{head 1}}
  \underbrace{
    \begin{bmatrix}
    c & d \\
    g & h \\
    k & l
    \end{bmatrix}
  }_{\text{head 2}}
  $$

  For example, $(3, 2, 1)$ corresponds to row 3, head 2, embedding dimension 1: element $k$ above.

  At this point, each head is an $n \times d$ matrix.

- We then transpose those to $(h, n, d)$. This doesn't change the shape or contents of the heads, it just changes how we index them. For example, $k$ would now be $(2, 3, 1)$.
- Now we calculate the attention weights $A$ as we did before.
  - The tensor libraries conceptually treat the first dimension ($h$, in our case) as a batching dimension. This means they do the matrix multiplication on each $h$ of the $(n, d)$ sub-matrices. The actual implementation is highly optimized.
  - Each $n \times d$ matrix becomes an $n \times n$ matrix, so the result is an $h \times n \times n$ tensor.
- We then multiply this by the multi-head value matrix $V_{h,n,d}$ to get an attention output $(h, n, d)$
- And finally, we transpose this back to $(n, h, d)$, reshape it back to $(n, d)$ and apply the $W_o$ projection.

These operations are highly optimized in the software that runs them, and down to the hardware level. In particular, the transposition operations don't move any actual data: they just change how the data gets indexed. This means they take essentially no time.

:::{note} Terminology note
In this section, I've been using $d$ for the per-head dimensionality, and $dh$ for the attention's overall dimensionality. This keeps things consistent with the terminology I've used throughout the book, which treats $d$ as the attention's conceptual dimensionality, and the multi-head configuration as an optimization.

Most LLM literature inverts this: it treats the multi-head dimensionality as the "real" one, labeled $d$ or $d_{model}$, and the per-head dimensionality as $\frac{d_{model}}{h}$.
:::

### FFNs

As I mentioned [in the previous chapter](#typical-ffn), each FFN in an LLM typically consists of the input sized $d$, one hidden layer sized $4d$, and an output layer sized $d$. The FFN's input and output correspond to a single token embedding; this gets evaluated [separately for each token](#ffn-output-shape), though GPUs and TPUs are able to do those separate evaluations efficiently in parallel.

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

After that, we just need to apply the activation function. This does have to be applied to each value separately, but GPUs and TPUs can efficiently parallelize that work.

This gives the full representation of each FFN layer:

$$
\text{Layer} = \text{Activation}( XW + B )
$$

To create the FFN, we just apply each layer serially.

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
