# Self-attention

:::{status} 1
:::

## What and _why_ is self-attention?

In [the previous chapter](./02-input-to-vectors), I described how to turn input text into a list of vectors. In the next section, we'll be using those vectors in a [feedforward network](#llm-components), which will make various inferences on them. But first, we're going to use a process called {dfn}`self-attention` to determine how each token draws information from the tokens around it.

:::{drawio} images/attention/llm-flow-self-attention
:alt: Self-attention sits between tokenization and the feedforward network
:::

:::{note} "Self attention" ⇄ "Attention"
The "self" in self-attention means each token attends to other tokens within the same input sequence.

In the context of LLMs, this is often shortened to just "attention".
:::

When I described the token embeddings in the previous chapter, I mentioned that they're combined with position embeddings to produce the final input embedding. This lets us differentiate between "have" as the first token in a sentence and "have" as the third token. This is a decent first step, but it's not enough: we want to know that it means something different in "we'll always {u}`have`" as compared to "Houston, we {u}`have`".

In other words, we want to learn what "have" means in the context of the specific sentence we see it in, factoring in the tokens around it. In the lingo of LLMs, we want to know how "have" {dfn}`attends to` each of those other tokens. This attention is the crucial innovation that GPT-style LLMs introduced over previous ML models.

Since the attention layer sits between the tokenization/embedding component and the feedforward network, I find it useful to be explicit about its inputs and outputs:

- The input is a vector with one element per token. Each element is itself represented by an input embedding (as described in the previous chapter), so this is an $n$-sized vector of $d$-sized vectors, where $n$ is the input size and $d$ is the embedding dimension.

  **tl;dr:** $n$ vectors of size $d$
- The output will be a vector of $n$ input-plus-attention elements, called an {dfn}`attention output`. Each attention output is a vector whose dimensionality I'll call $\delta$.

  **tl;dr:** $n$ vectors of size $\delta$

:::{aside}
I'll be using the following conventions throughout this chapter, and in general in the book:

- $n$: input size, in number of tokens
- $d$: dimensionality of each input token
- $\delta$: dimensionality of each output token

$d$ and $\delta$ are hyperparameters.
:::

:::{important} Two small but important notes

- $d$ and $\delta$ are technically independent hyperparameters, but in practice, $\delta = d$ in most LLMs. This allows us to stack multiple transformers in a way that makes the LLM much more powerful, as I'll describe in a later chapter.
- $\delta$ is _not_ a standard name in LLM literature. I'm picking it to show that it's conceptually different from $d$ while highlighting that in practice they're equal.

:::

In the rest of this chapter, I'll explain what this attention is concretely, and how we compute it. I'll start by building an intuition and motivation around what we're building, and then go into the details of how it works. Finally, the last part of this chapter will introduce some important real-world refinements.

:::{attention} Hang onto your butts!
I found this hard to wrap my head around. I've tried to interweave the "what" of the computations with the "why" of why they work as best I can. Be patient with yourself: this is tricky, but it's crucial to understanding how LLMs work.
:::

We're going to be making extensive use of matrix math in this chapter. Make sure you remember how that works, and in particular the shapes of the matrices when they're multiplied. It's covered in the [earlier chapter on matrix math][math].

[math]: ../01-Introduction/03-vectors-matrices-tensors.md

````{seealso} Quick refresher, if you need it
:class: dropdown

```{embed} #matrix-math-summary
```

````

## Building a high-level intuition of what we need

### What if we had infinite compute power?

As I mentioned above, what we really want to answer is: "for every input token, what information does it draw from every other token?" That question has nuance, and as you'll recall from [the earlier overview](#vectors-are-nuance), nuance means vectors --- or matrices.

If we have $n$ input embeddings, and each can attend to every other (including itself), we can visualize attention as an $n \times n$ grid:

:::{drawio} images/attention/attention-weights-houston
:alt: 5 by 5 attention weight grid with "Houston we have a problem" as both rows and columns. Each cell shows how the row word attends to the column word.
:::

So, what's in each of these cells?

Each cell tells us what input token A draws from input token B. Since that translates an input $d$-vector (the input embedding) into an output $\delta$-vector (the attention output), we can use a $d \times \delta$ matrix:

$$
\text{Input}_{1 \times d}
\cdot \underbrace{\text{Transformation}_{d \times \delta}}
= Output_{1 \times \delta}
$$

That means each element in the grid has to be a $d \times \delta$ matrix:

:::{drawio} images/attention/attention-weights-houston-vectors
:alt: n-by-n grid, where each cell is a delta-sized vector
:::

The problem is that this is an $n \times n \times d \times \delta$ tensor, which would be far too large to reasonably store and train on. Worse yet, it grows by the square of the input size! Ideally, we'd like to have something that only grows as a function of the $d$ hyperparameter.

:::{note} How big are we talking?
:class: dropdown
:open: false
In a modern LLM, the embedding dimensions $d$ are roughly 4k - 10k, and the context lengths $n$ vary widely, from about 4k to 200k and up. If we assume $n = 8k$ and $d = 4k$ (we'll assume $d = \delta$), and a 4-byte float per value in the tensor, this gives us:

$$
\begin{align}
& (n \times n \times d \times d) \times 4 \text{ bytes} \\
= & (8192 \times 8192 \times 4096 \times 4096) \times 4 \text{ bytes} \\
\approx & (1.1 \times 10^{15} \text{ parameters}) \times 4 \text{ bytes} \\
\approx & 4.5 \text{ PB}
\end{align}
$$
:::

### Focusing our attention

Instead of trying to learn everything about the relationships between inputs, we'll have the attention layer focus on just one or two kinds of relationships. For example, an attention layer may focus on understanding how parentheticals fit into a sentence, or learning about subject-verb agreement.

Let's take subject-verb agreement as an example. Of course, not all tokens are nouns or verbs. To learn subject-verb agreement, the attention mechanism needs to first focus on subjects-verb pairs, and mostly ignore the others. Otherwise, the model will train on noise, or even contradictory information. For example, the suffix "-s" usually marks singular verbs but plural nouns, and conjunctions don't have the concept of pluralization at all.

Once it finds the relevant token pairs, the attention mechanism needs to extract the information. For example, a layer learning subject-verb agreement needs to extract whether the subject is singular or plural.

In practice, this information is almost never neatly packaged into a single dimension in the token's embedding; it's spread out and entangled across several dimensions. So, the attention layer needs to learn how to extract and recombine those distributed properties into a useful representation.

With all that, now we have a more tractable problem than just "find the relationships between tokens". We need to:

1. Learn which token pairings matter to the specific relationship that this layer is learning.
2. Learn how to extract and combine the relevant parts of the token embeddings to produce the right output

:::{tip} Only learning one relationship?
For purposes of understanding attention, you can think of the layer learning just one relationship, as I just described above. Of course, languages have _lots_ of relationships. As we'll see later in this chapter, and then again in [](./05-putting-it-together.md), our LLM will actually have many instances of attention layers (called "heads"). Each will focus on one relationship, but because they all start with different random initial values, each will randomly converge towards a different relationship.

What the relationships _actually_ are can be very opaque. When researchers look at the learned parameters, some layers seem pretty straightforward, like learning "words depend on the word before them". Other layers seem to conflate a few seemingly unrelated relationships, and still others are totally incomprehensible. Gaining insight into what these layers mean is an area of active research.
:::

Let's take a look at what structures could let us answer these two questions.

### Breaking down the problem

As I covered above, instead of asking generally "how does input A attend to input B", we'll approximate that question by asking two simpler ones:

- How much does input A care about input B (for this layer's relationship)?
- How should input B express its information (for this layer's relationship)?

Note that our two questions involved three usages of input tokens:

- How much does **(input A)** care about **(input B)**?
- How should **(input B)** express its information?

The LLM needs to learn something about each of these usages, so we'll use a learned transformation for each one. We'll call these transformations $W_q$, $W_k$, and $W_v$ (you'll see why in just a moment). Now we have:

- How much does **(input A transformed by $W_q$)** care about **(input B transformed by $W_k$)**?
- How should **(input B transformed by $W_v$)** express its information?

Let's think about what shapes these transformations should be.

The "how much" question is a scalar (think, "on a scale from 1 to 100, how much does...?"). We have two input tokens in this question, so if we turned each into a vector of equal length, we could calculate their dot product to get that scalar. Let's translate them into $\delta$-sized vectors, which we can do via a $d \times \delta$ matrix:

$$
\underbrace{Input}_{1 \times d} \cdot \underbrace{Transformation}_{d \times \delta} = \underbrace{Output}_{1 \times \delta}
$$

Since we have two separate transformations --- one for input A and one for input B --- we'll define two such matrices:

$$
\begin{align}
W_q & \Rightarrow d \times \delta \\
W_k & \Rightarrow d \times \delta \\
\end{align}
$$

Note that while "how much" score is a single number, the transformation matrices that produce it encode quite a bit of nuance: each has $d \times \delta$ learned parameters!

:::{tip} Why dot products?
:class: dropdown
So, we took two $d$-vectors, transformed them into two $\delta$ vectors, and then took their dot product to get a scalar. But why use a dot product?

Geometrically, dot products measure how aligned two vectors are _if_ those vectors are normalized to have the same magnitude ("length", basically). But our input embeddings aren't normalized, so that doesn't apply here.

The real answer, as far as I can tell, is that dot products are efficient to compute, and they work well in practice. They are also differentiable, which will be important for training.

Note that there's no reason for the two transformed vectors to be $\delta$ specifically; they could in principle be any size, as long as the two sizes are equal (so that we can dot-product them). This means that $W_q$ and $W_k$ could each be $d \times x$ for any $x$. Most LLMs use $\delta$, and we'll assume that, to keep things simple.
:::

The answer to "how should it express its information" is different --- we're not asking for a score, but for the actual content. We'll represent this output as a $\delta$-dimensional vector (which will determine our attention output's dimension). Again, this means we need a $d \times \delta$ matrix to transform the input $d$-vector to a $\delta$-vector:

$$
W_v \Rightarrow d \times \delta
$$

Crucially, because this approach is just an approximation of the $n \times n \times d \times \delta$ matrix, we don't need a separate set of $W_q$ / $W_k$ / $W_v$ for each cell in the $n \times n$ grid. Instead, we just have three weights total for the whole attention layer: one $W_q$, one $W_k$, and one $W_v$.

:::{important} Recap

- Our goal is to turn each input $d$-vector into an output $\delta$-vector, in a way that encodes the relationships between each pair of tokens.
- The naive way to do this would be a $d \times \delta$ matrix for each of these pairs (there are $n \times n$ of them), but this is too huge.
- Instead, we decompose the problem into two smaller problems, which we can answer using three $d \times \delta$ matrices.

:::

The "query / key / value" terminology comes from an analogy to database lookups:

- $W_q$ ("query weight matrix"): Turns A into a query.
- $W_k$ ("key weight matrix"): Turns B into a key for the query to match against.
- $W_v$ ("value weight matrix"): Turns B into a value that the query returns.

:::{tip} Alternative mental model
:class: dropdown

The $W_q$ and $W_k$ distinction didn't really make sense to me, in part because it feels arbitrary. Mathematically, you could swap the two names and everything still works: both weight matrices are used to transform the input embeddings into $\delta$-vectors, which then get dot-producted; and dot products are commutative.

To me, a better mental model is to consider the $(A, B)$ pair as a single, composite key; and then have weight matrices $W_{k1}$ and $W_{k2}$ to transform its components into $\delta$-vectors to before calculating their dot products.

I only mention this in case the $W_q$ / $W_k$ division also trips you up. That is the standard terminology, though, so it's what I'll be using throughout this book.
:::

(computing-attention-weights)=

## Computing attention weights

The next sections will describe the mechanics of calculating attention. If the above doesn't make sense, it may be useful to move on for now, and then re-read it once you understand how the weight matrices actually get used.

### Overview

For each token within the input, we'll focus on that token and do a bunch of calculations centered on it. We'll call that token the {dfn}`query token`: it's the token for which we want to ask (that is, "query") how it attends to each token in the input.

With that query token in mind, we'll look at each token in the input, treating each one in turn as a {dfn}`key token`.

1. First, calculate the query token's {dfn}`query vector`, using $W_q$ and a bias called $b_q$ (more in this below).
2. Then calculate {dfn}`attention scores` for each key token. These are scalars that tell us how much the query token should care about each key token. There is one per key token. To calculate these scores:
    1. Use $W_k$ to calculate the {dfn}`key vector` per key. We'll also add a bias, called $b_k$, just as we did for the query vector.
    2. Take the dot product of the query and key vectors to get the attention score for each key.
3. Next, normalize these attention scores into {dfn}`attention weights` (still one scalar per key token).
4. Next, compute {dfn}`value vectors` for each key, weighted by their respective attention weights:
    1. Use $W_v$ and a $b_v$ bias to transform each key into a $\delta$-vector
    2. Multiply each of those $\delta$-vectors by its respective attention weight to compute the weighted value vectors, again one per key.
5. Finally, sum the weighted values to get the {dfn}`context vector`, which is the output for this query token. Since this is the sum of $\delta$-vectors, it is also a $\delta$-vector.

:::{aside}

- **$W_q$, $W_k$, $W_v$**: learned parameter matrices; each has size $d \times \delta$, where $d$ and $\delta$ are both hyperparameters, and typically $d = \delta$
- $b_q$, $b_k$, $b_v$: learned parameter vectors, each $\delta$ elements.
- {dfn}`query, key, value vectors`: $\delta$-sized vector activations based on the input embeddings and the weight matrices
- {dfn}`attention score, attention weights`: scalar activations based on the query and key vectors
- {dfn}`weighted values`: $\delta$-sized vector activations based on the attention weights and input embeddings
- {dfn}`context vectors`: $\delta$-sized vector activations based on the attention weights
:::

(attention-overview-graphic)=
:::{drawio} images/attention/overview
:alt: visual representation of the overall flow described above
:::

Again, all of that work is just for a single query token. We'll repeat it for each token in the input to produce $n$ context vectors of size $\delta$. This is our attention layer's output.

:::{important}

- In contrast to most other [fundamental concepts](#conceptual-layers) in LLMs, the attention parameters are matrices, not vectors. (This is because what we're learning is how to transform vectors.)
- There is only _one_ $W_q$, one $W_k$, and one $W_v$. There isn't a different $W_q$ per query token, for example. $W_q$ essentially encodes what it means to be a query token, independent of any particular embedding; and similarly for $W_k$ and $W_v$.
- The attention weights themselves are [activations](#parameter-vs-activation) based on the specific prompt that the LLM is looking at.
:::

Let's walk through the specifics.

### $W_q$ and $b_q$ → query vector

::::{aside}
:::{drawio} images/attention/llm-flow-self-attention-query
:alt: query token times Wq = query vector
:::

:::{div}
:class: text-center aside-caption
[Step 1](#attention-overview-graphic)
:::

::::

This just transforms the query token by the weight matrix $W_q$:

- We start with the query token's input embedding, which is a vector of size $d$
- We have the query weight matrix $W_q$, which is $d \times \delta$
- We multiply them together, and the result is a vector of size $\delta$:
  $$
  \underbrace{query\ token}_{1 \times d} \cdot \underbrace{W_q}_{d \times \delta} = \underbrace{query\ vector}_{1 \times \delta}
  $$
- We add the bias $b_q$. This is a vector of size $\delta$, so we just add the two vectors' corresponding elements.

That's it!

Why the bias? We can think of the query vector as analogous to a linear function:

$$
\text{query vector} = XW_q + b_q \Longleftrightarrow ax + b
$$

Just as you'd need the $b$ coefficient to match data to a linear function in standard Cartesian math, you need $b_q$ to match the input to the query vector.

### Query vector, $W_k$ and $b_k$ → attention scores

::::{aside}
:::{drawio} images/attention/llm-flow-self-attention-score
:alt: query tokens and input tokens turn into attention scores
:::

:::{div}
:class: text-center aside-caption
[Step 2](#attention-overview-graphic)
:::

::::

This step happens for each key token (that is, each embedding in the input vector).

First, we'll calculate the key vector for each key token. This is a $\delta$-sized vector. Similar to how we calculated the query vector, this is just $key\ embedding \cdot W_k + b_k$.

Now we have two $\delta$-sized vectors: the query (from the previous step) and the key. We compute their dot-product-math to combine them into a scalar.

We call this dot product the raw {dfn}`attention score` for this key.

### Attention scores → attention weights

::::{aside}
:::{drawio} images/attention/llm-flow-self-attention-weight
:alt: attention scores normalize into attention weights
:::

:::{div}
:class: text-center aside-caption
[Step 3](#attention-overview-graphic)
:::
::::

At this point, we have $n$ raw attention scores, each corresponding to a key token.

These scores can be all over the place --- positive, negative, and at vastly different scales --- so we normalize them to a probability distribution. This distribution is an $n$-vector called the {dfn}`attention weights`. Its values are all between 0 and 1, and they all sum to 1.

Normalizing the attention scores to attention weights improves the learning process by making the attention more differentiable and keeping the scales of the values more stable.

This normalization happens in two steps:

1. First, we divide each attention score by $\sqrt{\delta}$ (the square root of the output embedding size)
2. Then, we apply a function called softmax, which takes a vector of scalars and normalizes them to a probability distribution.

I'll explain these backwards: first softmax, then the scaling.

Softmax is a function that converts a vector of numbers into a probability distribution. You don't actually need to know its definition, but what _is_ important is that it's sensitive to the scale of its inputs: The larger the scale, the more softmax magnifies differences in probabilities.

:::{tip} Softmax details
:class: dropdown
:open: false

Softmax is defined as:

$$
\text{softmax}(v)_i = \sigma(v)_i = \frac{e^{v_i}}{\sum_{j} e^{v_j}}
$$

In English: each element $v_i$ is transformed into its exponential ($e^{v_i}$) and then divided by the sum of all the exponentials.

The exponential function $e^x$ is very sensitive to its inputs, and in particular, large numbers cause it to change a lot. This means that if two numbers are 10% different, the exponential function and thus softmax will react a lot more if they're bigger. For example, here are three vectors that each have two numbers 10% from each other:

$$
\begin{array}{lll}
\sigma( \, [ \, 1.0, & 1.1 & ] ) = [ \, 0.475, 0.525 \, ] \\
\sigma( \, [ \, 10,  & 11  & ] ) = [ \, 0.269, 0.731 \, ] \\
\sigma( \, [ \, 100, & 110 & ] ) = [ \, 0.000045, 0.999955 \, ]
\end{array}
$$

You can play around with [a visualization of softmax](../static/softmax.html) to get a sense of how it works.
:::

To keep softmax from becoming too extreme, we first divide the attention scores by $\sqrt{\delta}$. This factor comes from statistics. Remember that the raw attention score is a dot product that's the sum of $\delta$ terms, one per dimension. These terms are roughly independent, so the standard deviation of their sum grows as $\sqrt{\delta}$ (this is standard statistics, which we don't need to get into the details of here). By dividing by $\sqrt{\delta}$, we keep the typical magnitude of attention scores consistent regardless of $\delta$. This ensures that softmax operates in a reasonable range, and doesn't get thrown off by large scales.

In other words, as $\delta$ grows, so do the dot products' variance. This growth happens by a factor of $\sqrt{\delta}$, and left unchecked it would cause softmax to lose nuance between values that are actually fairly close. Dividing by $\sqrt{\delta}$ lets softmax keep that nuance. Note that I wrote above that the terms are roughly independent, but of course they're not _actually_ independent: the whole point of training is to find patterns in them. Still, the $\sqrt{\delta}$ scaling has empirically been found to work, so that's what people use.

This "scaling plus softmax" is called, appropriately enough, the scaled dot-product attention. When it's applied to the raw attention scores we calculated earlier, the result is the normalized {dfn}`attention weights`.

:::{important} Attention scores vs attention weights
These two terms are similar, but it's crucial to keep them separate.

attention scores
: "Raw" outputs from the combination of query vector and key vector. These can be pretty much any value, but they're only used to calculate the attention weights.

attention weights
: Normalized values that represent the percentage of attention that each token gets. These are all between 0 and 1, and they sum to 1.
:::

### $W_v$ and $b_v$ → weighted value vectors

::::{aside}
:::{drawio} images/attention/llm-flow-self-attention-weighted-value
:alt: attention weights combine with key embeddings to form the vector vectors
:::

:::{div}
:class: text-center aside-caption
[Step 4](#attention-overview-graphic)
:::
::::

All of the work until now has been to calculate the attention weights, which are an $n$-sized vector of scalars that answer the first component of attention: "for each input A, how much does it care about input B?" Now we'll answer the second component: how should input B express its information?

We'll start with familiar ground, by turning our $d$-sized key embeddings into $\delta$-sized vectors by multiplying them by a weight matrix. This time we'll use the $W_v$ weight matrix and $b_v$ bias, and the result is a {dfn}`value vector`. As with the key vector, we have one such value vector per input token.

From here, we calculate intermediate "weighted values" by multiplying each value vector by its corresponding attention weight. For example, let's say:

- Attention weight $4$ is 0.27. This means that 27% of the query token's attention is devoted to input 4.
- Input token 4's value vector is $[6.2, 1.4, 7.9]$ (here we have $\delta = 3$).

In this case, the weighted value vector for input 4 is:

$$
\begin{align}
  & 0.27 \cdot [6.2, 1.4, 7.9] \\
= \; & [(0.27 \cdot 6.2), (0.27 \cdot 1.4), (0.27 \cdot 7.9)] \\
= \; & [1.67, 0.378, 2.13]
\end{align}
$$

We do this calculation for each key embedding to get $n$ weighted value vectors.

### weighted value vectors → context vector

::::{aside}
:::{drawio} images/attention/llm-flow-self-attention-context
:alt: attention weights combine with values to form the context vector
:::

:::{div}
:class: text-center aside-caption
[Step 5](#attention-overview-graphic)
:::
::::

At this point, we have $n$ weighted values. Each is a $\delta$-sized vector that represents its respective key embedding, scaled by how much the query token cares about it and projected to represent what that key means in the context of the relationship this attention layer is learning.

We simply [sum those vectors](#adding-matrices) to get the {dfn}`context vector`, which is also a $\delta$-sized vector. This represents the attention layer's output for this query token.

### Repeat to get the full layer output

Recall that all of this happened from the perspective of a single input, which we called the query token. For this token, we:

1. Calculated a query vector ($query\ token \cdot W_q$).
2. Dot-producted that query vector against every input's key vector ($key\ token \cdot W_k$) to get, for each input, a raw attention score.
3. Normalized those attention scores into probabilities, one per input; we called these the attention weights.
4. Applied each of those probabilities to a corresponding value vector ($key\ token \cdot W_v$) to get weighted value vectors, one per input
5. Summed up those weighted value vectors to get a single context vector

All of this gives us the $\delta$-dimensional context vector for that one query token.

We then repeat this for each of the $n$ inputs, treating each as the query token in turn. The result is our attention layer's output: the full {dfn}`attention output matrix`, or just {dfn}`attention output` for short. This has one context vector for each input, so it's $n$ vectors of size $\delta$.

:::{drawio} images/attention/llm-flow-self-attention-output-matrix
:alt: attention weights combine with values to form the context vector
:::

## Causal attention mask

In all of the above, we've been calculating the full $n \times n$ attention grid, as we saw above:

:::{drawio} images/attention/attention-weights-houston
:alt: the full n times n attention grid
:::

The problem is that at inference, we're going to be predicting one token at a time. This means that when we predict token, we won't yet know the tokens after it --- and thus can't know how they'll attend to it:

(causal-attention-grid)=
:::{drawio} images/attention/causal-attention-grid
:alt: The same n times n grid as before, but with the top-right crossed out to show that we don't know those pairs.
:::

To account for this, we'll fill the top-right portion of the attention scores with $-\infty$, right before applying softmax. When we apply softmax to each row, the $-\infty$ will turn to 0, and they'll be disregarded as we normalize the rest of the values.

This triangle of $-\infty$s is called the {dfn}`causal attention mask`.

## Real-world improvements

The above covers the fundamental aspects of how self-attention works, but there are several crucial ways that it's augmented in real-world LLMs. Don't worry: the hardest part is behind us. Still, it's important to know about these if you want to understand how real LLMs work.

(multi-head)=

### Multi-head attention and $W_o$

When I wrote above that there's only one each of $W_q$, $W_k$, and $W_v$, that was a bit of a simplification. Everything I've described above --- the weight matrices, vectors, etc --- forms a unit called an {dfn}`attention head`.

The problem is that a single attention head can get somewhat myopic, focusing primarily on just one aspect of the input tokens. For example, a head may end up focusing just on semantic interactions between tokens, or just on their grammatical relationships. (The actual relationships it learns are more abstract than that, but I'm "translating" the properties it learns into more intuitive relationships).

To solve this, LLMs actually use multiple heads, each with their own $W_q$ / $W_k$, / $W_v$ matrices and biases. Each one of these heads acts independently, finding its own relationship to learn.

In this {dfn}`multi-head` arrangement, each head's output has $\frac{\delta}{h}$ dimensions, where $\delta$ is the attention layer output's dimensionality (as we've been using it all along) and $h$ is the number of heads. For example, if we want the attention output to have 720 dimensions, and we want 12 heads (these are both hyperparameters the model designer picks), each head would have dimensionality 60. This then determines how big each head's weight matrices are: each will be $d \times \frac{\delta}{h}$.

:::{aside}

- **$h$**: hyperparameter; the number of heads in a multi-head attention
:::

Each head's output is $n$ rows of size $\frac{\delta}{h}$, and we can think these as $n \times \frac{\delta}{h}$ matrices. We then concatenate them to get our desired shape, an $n \times \delta$ matrix.

(multi-head-scaling)=
The only gotcha in this process is that when we scale down the attention scores, we now have to scale them down by $\sqrt{d/h}$ instead of $\sqrt{d}$. Remember that the reason we did that scaling was to account for the dot products growing as the embedding dimension grows; but within each head, that embedding dimension is now $\frac{d}{h}$.

You may be thinking that it seems odd to just concatenate matrices that don't necessarily have much to do with each other, and the borders of which are essentially "jumps" between differently-learned relationships. How would the layers that consume this matrix know how to make sense of them and combine them into a single, coherent input?

(w-o-projection)=

To solve that problem, multi-head models introduce one more matrix, $W_o$ (for "output"), along with its bias. This is a $\delta \times \delta$ learned matrix that encodes how to combine all the heads into a single, appropriately blended result.

$$
\underbrace{concatenated\ heads}_{n \times \delta}
\cdot \underbrace{W_o}_{\delta \times \delta} + b_o
= \underbrace{layer\ output}_{n \times \delta}
$$

:::{drawio} images/attention/multi-head
:alt: In multi-head attention, each head produces its own output, and the W_o matrix combines them into a single output for the layer
:::

### Multiple layers

In all of the above, we've been talking about "the" self-attention layer, as if there's only one. In practice, an LLM will have many attention layers.

In the [next section](./04-feedforward-network), I'll describe the LLM's feedforward network, which makes inferences about the attention output matrix we've been developing in this chapter. The attention layer and feedforward network together form a {dfn}`transformer block`. Modern LLMs stack several of these blocks together, with each block's output feeding into the next's attention.

I'll describe this in more detail in [](./05-putting-it-together.md). For now, just know that the description of "the" attention feeding into "the" feedforward network is a simplification.

### RoPE

As I mentioned in the previous chapter, modern LLMs don't add positional encoding to the input embeddings. Instead, they use something called RoPE, which gets applied in the attention layer.

I won't describe RoPE in this book --- sorry! Just know that it's the modern replacement for positional embeddings.

### Dropout

There's one more aspect of attention, which is something called dropout.

This aren't a part of the attention's fundamental architecture: it's only applied at training. As such, I'm going to put it off until the later chapters on training.

If you haven't heard about dropout yet, you can forget I mentioned it. I only bring them up in case you're also using another resource (another book, or asking an LLM questions) and it mentions it. It's often taught as part of attention, but I think they're best held off until we give training its full treatment.

## "The context is full"

If you've used LLMs, you may have heard about "the context" as an almost mythical thing to be kept safe. The context can't get too full; you can't let it get too confused with bad prompts or intermediate results; some parts of it belong to the tooling and some belong to you.

If you read about "context vectors" above and wondered if these are related: good news, they are! In fact, you now have enough to build a solid understanding of what this all-important context _is_.

In short, "the context is full" means that the input is as long as the LLM will allow. This is primarily driven by two factors:

- **The attention score and weight matrices grow quadratically with input length.** For each query token, we compute $n$ attention scores and weights (one per key token). Across all $n$ query tokens, this gives us an $n \times n$ matrix of scores and another $n \times n$ matrix of weights.
  - This means that the memory and processing the LLM needs grows as the square of the input length.
  - We have one set of these matrices per head, so with 8 - 96 heads per layer, and multiple layers per model, this really adds up!
- **Training on longer sequences is expensive.** Recall the $n \times n$ attention grid where each cell represents the relationship between two tokens. For the model to make useful inferences about a token pair $X$ tokens apart, it needs to have been trained on data with at least $X$ tokens --- and as the above bullet point mentions, this gets expensive quickly.

LLM designers must balance the cost of training on long sequences against the usefulness of longer context windows.

Note that the learned weight matrices ($W_q$, $W_k$, $W_v$, and $W_o$) are _not_ what limits context length. These matrices have fixed sizes based purely on the model's hyperparameters ($d$ and $\delta$), not on input length.

## Next up

As I've mentioned already, attention is the first part of an LLM's transformer block. In the next chapters, I'll explain the second part of the transformer block --- the feedforward network --- and then how all the pieces fit together to form a full LLM.
