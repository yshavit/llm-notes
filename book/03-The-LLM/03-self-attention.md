---
title: Self-attention
downloads:
  - file: softmax.html
    title: Interactive Softmax
---

## What and _why_ is self-attention?

In [the previous section](./02-input-to-vectors), I described how to turn input text into a list of vectors. In the next section, we'll be feeding those vectors into a neural network. But first, in this section, we're going to use a process called {dfn}`self-attention` to determine how each word in the input affects the other words in the input.

{drawio}`Self-attention sits between tokenization and the neural net|images/05/llm-flow-self-attention`

This piece of the model is the crucial innovation that GPT-style LLMs introduced over previous ML models. It lets the LLM encode not just what any one word means, but what each word means in the context of the _specific_ words that precede before it. It's not enough to know what "have" means, or even what it means as the third word of a sentence: we want to know that it means something different in "we'll always {u}`have`" as compared to "Houston, we {u}`have`".

{drawio}`5 by 5 attention weight grid with "Houston we have a problem" as both rows and columns. Each cell shows how the row word attends to the column word.|images/05/attention-weights-houston`

Since this layer sits between the tokenization/embedding component and the neural network, I find it useful to be explicit about its inputs and outputs:

- The input is a vector representing each token in the input. Each token is itself represented by an input embedding (as described in the previous chapter), so this is an $n \times d$ matrix, where $n$ is the input size and $d$ is the dimensionality of each embedding.
- The output will be a vector of these attentions outputs. Each attention output is a vector. We'll call that vector's dimensionality $\delta$, so the ultimate output will be a $n \times \delta$ matrix.
  :::{important} Two small but important notes
  - $d$ and $\delta$ are technically independent [hyperparameters](#parameter-vs-activation), but in practice, $\delta = d$ in most LLMs. That's because if $\delta$ were smaller than $d$, you'd be losing information; and if it were bigger, you'd be encoding more information than you started with. It'd be like writing a number to 10 decimal places when your ruler only measures precisely to 2.
  - $\delta$ is _not_ a standard name in LLM literature. I'm picking it to show that it's conceptually different from $d$ while highlighting that in practice they're equal.
  :::

To do this, we're going to first build up the basics of an attention output. Then, we'll introduce some important real-world refinements.

## Computing attention outputs

:::{attention} Hang onto your butts!
I found this all very hard to wrap my head around. I've tried to interweave the "what" of the computations with the "why" of why they work as best I can. Be patient with yourself: this is tricky, but it's crucial to understanding how LLMs work.
:::

### Building a high-level intuition of what we need

As I mentioned above, what we really want to answer is: "for every input, what does it mean with respect to every other input?" That question has nuance, and as you'll recall from [my overview](#vectors-are-nuance), nuance means vectors.

The naive approach is conceptually easy: we need an $n \times n$ grid, where each item answers "what does A mean with respect to B?" In other words, each cell in that grid translates the input $d$-vectors into output $\delta$-vectors.

Remember from the [matrix multiplication](#matrix-multiplication-notes) chapter that $A_{a \times b} \cdot B_{b \times c} = C_{a \times c}$. (I'll be using this property a lot in this chapter, so make sure you internalize it!)

This means each element in the grid has to be a $d \times \delta$ matrix:

{drawio}`n-by-n grid, where each cell is a delta-sized vector|images/05/attention-weights-houston-vectors`

The problem is that this is an $n \times n \times d \times \delta$ tensor, which would be far too large to reasonably store and train on.

:::{note} How big are we talking?
:class: dropdown
:open: false
In a modern LLM, the embedding dimensions are roughly 4k - 10k, and the context lengths vary widely, from about 4k to 200k and up. If we assume $n = 8k$ and $d = 4k$ (we'll assume $d = \delta$), and a 4-byte float per value in the tensor, this gives us:

$$
\begin{align}
& (n \times n \times d \times d) \times 4 \text{bytes} \\
= & (8192 \times 8192 \times 4096 \times 4096) \times 4 \text{ bytes} \\
\approx & (1.1 \times 10^{15} \text{ parameters}) \times 4 \text{ bytes} \\
\approx & 4.5 \text{ PB}
\end{align}
$$
:::

So, instead of asking "how does input A affect input B", we'll approximate that question by asking two simpler ones:

- How _much_ does input A care about input B?
- What does input B mean?

The first of those we'll represent with a simple scalar; the second is a nuanced question, so we'll represent it as a ($\delta$-dimensional) vector.

Note that our two questions involved three usages of tokens:

- How much does **(input A)** care about **(input B)**?
- What does **input B** mean?

Each of these usages has some nuance, so we'll use a learned transformation for them. We'll call these $W_q$, $W_k$, and $W_v$ (you'll see why in just a moment). Now we have:

- How much does **(input A transformed by $W_q$)** care about **(input B transformed by $W_k$)**?
- What does **(input B transformed by $W_v$)** mean?

We want to represent each of these as a $\delta$-dimensional vector, which we derive from the $d$-dimensional input embedding and the respective $W_\star$ weights. Remembering that we can transform a $d$-vector to a $\delta$ vector using a $d \times \delta$ matrix, this means each weight is a $d \times \delta$ matrix:

$$
Input_{(1 \times d)} \cdot W\star_{(d \times \delta)} = Output_{(1 \times \delta)}
$$

The "query / key / value" terminology comes from an analogy to database lookups, though I personally don't find the analogy useful. In case you do, the idea is:

- $W_q$ ("query weights"): What am I looking for?
- $W_k$ ("key weights"): What do I offer as context?
- $W_v$ ("value weights"): What information should I pass along?

:::{tip} Putting it all together
In trying to understand this, I kept struggling to understand why we needed these various transformations. After all, the input embeddings we previously computed already contain a concept of nuance. What more do these transformations add?

Remember that our ideal transformation would be an $n \times n \times d \times \delta$ tensor, but that's too big. We're essentially using a couple tricks to compress it.

- First, we're replacing the inner $d \times \delta$ matrices with a 0-dimensional scalar that just represents how _much_ (as oppoed to more generally "how") each input affects the other.
- Then, we're applying this scalar against a vector that represents what each input embedding means _in the context of attention_.

To translate the input embeddings to "the input embedding in the context of attention", we add the $W_v$ matrix. This matrix essentially encodes the "in the context of attention" learnings. The $W_q$ and $W_k$ matrices similarly encode "how much does A-in-attention-context affect B-in-attention-context."

The math would still work if we did without these matrices and just did something like a dot product of A and B for the "how much" question, and used the B embeddings directly for "what does B mean". (We'd need to require $d = \delta$, but that wouldn't be a show-stopper.) But this would compress the attention _too_ much, and the attention layer would be hobbled.

Adding the three weight matrices gives us a happy medium: a mere $3 \times (d \times \delta \times 4 \text{ bytes})$ gives us a useful approximation of the ideal tensor at a tiny fraction of the cost (hundreds of megabytes, instead of petabytes).
:::

The next sections will describe how these matrices fit together. If the above doesn't make sense, it may be useful to move on for now, and then re-read it once you understand the mechanics of how we use these weight matrices.

### Overview of how the weights fit together

For each token within the input, we'll focus on that input and do a bunch of calculations centered on it. Let's call that token the query token: it's the token for which we want to ask (that is, "query") how it attends to each token in the input.

I'll start with an overview of the process, and then the next sections will go into details on each step.

1. First, we apply each of the weight matrices:
    1. We apply the query weights ($W_q$) to the query token to get its {dfn}`query vector`: what is this token looking for?
    2. We apply the key weights ($W_k$) to every token to get the {dfn}`key vectors`: how can we match against each token?
    3. We apply the value weights ($W_v$) to every token to get the {dfn}`value vectors`: what information does each token contain?
2. Then we combine the query with each key to compute {dfn}`attention scores`, one per token. These are scalars that tell us how much the query token should care about each other token.
3. Then we normalize these attention scores into {dfn}`attention weights` (are still one scalar per token).
4. Then we combine the attention weights with each value to compute {dfn}`weighted values`, again one per token. These are vectors.
5. Finally, we combine the weighted values to get the {dfn}`context vector`: a weighted blend of information from all tokens, focused on what's relevant to the query token.

{drawio}`visual representation of the overall flow described above|images/self-attention/overview`

Again, all of that work is just for a single query token. We'll repeat it for each token in the input to produce a context vector per token. The final result is a vector of vectors, which we'll represent as a matrix. This matrix is our self-attention layer's output.

:::{important}

- In contrast to most other [fundamental concepts](#conceptual-layers) in LLMs, the attention parameters really are matrices, not vectors.
- There is only _one_ $W_q$, one $W_k$, and one $W_v$. Each matrix gets paired with various tokens as we'll see below, and its training will reflect that. But there isn't a different $W_q$ per query token, for example. $W_q$ essentially encodes what it means to be a query token, independent of any particular embedding; and similarly for $W_k$ and $W_v$.
- The attention weights themselves are [activations](#parameter-vs-activation) based on the specific prompt that the LLM is looking at.
:::

Let's walk through the specifics.

### $W_q$ 🡒 query vector

:::{aside}
{drawio}`query token times Wq = query vector|images/05/llm-flow-self-attention-query`
:::

This one is easy.

- We start with the query token's input embedding, which is a vector of size $d$
- We have the query weight matrix $W_q$, of size $d \times \delta$
- We just multiply them together: $query = embedding \cdot W_q$
- Out comes and we get a vector of size $\delta$

That's it! We do this just once per query input.

### Query vector and $W_k$ 🡒 attention scores

:::{aside}
{drawio}`query tokens and input tokens turn into attention scores|images/05/llm-flow-self-attention-score`
:::

We'll do this step once for every input token; I'll call these tokens the key tokens (though that's not a standard term).

First, we'll calculate the key for each key token, which is a $\delta$-sized vector. Similar to how we computed the query vector, this key vector is just $key\ embedding \times W_k$.

Now we have two $\delta$-sized vectors: the query (from the previous step) and the key. We can compute their [dot product](#dot-product-math) to combine them into a scalar.

:::{note} Why dot products?
Geometrically, a dot product represents how aligned two vectors are, assuming the vectors are normalized. Our query and key vectors have _not_ been normalized; but over training, the $W_q$ and $W_k$ matrices' values will converge to capture meaningful relationships.

In other words, the dot product of these two vectors represents how aligned they are; or, in LLM terms, how much the query "makes sense" relative to the key. This tells us how much the query token cares about the input token that corresponds to this key.
:::

We call this dot product the raw {dfn}`attention score` for this key.

### Attention scores 🡒 attention weights

:::{aside}
{drawio}`attention scores normalize into attention weights|images/05/llm-flow-self-attention-weight`
:::

Since we repeat the attention score process for each token in the input, we end up with an $n$-vector of attention scores. These scores can be all over the place — positive, negative, and at vastly different scales — so we normalize them to a probability distribution, which we call the {dfn}`attention weights`. The values within this distribution  are all between 0 and 1, and they all sum to 1.

Normalizing the attention scores to attention weights improves the learning process by making the attention more differentiable and keeping the scales of the values more stable.

To normalize the values, we use two functions:

1. First, we scale the weights by $\sqrt{\delta}$ (the square root of the input embedding)
2. Then, we apply a function called softmax, which takes a vector of values and normalizes them to a probability distribution.

I'll explain these backwards: first softmax, then the scaling.

Softmax is a function that converts a vector of numbers into a probability distribution. You don't actually need to know its definition, but what _is_ important is that it's sensitive to the scale of its inputs: The larger the scale, the more softmax magnifies differences in probabilities.

:::{tip} Softmax details (very optional)
:class: dropdown
:open: false

[Softmax] is defined as:

$$
\text{softmax}(v)_i = \sigma(v)_i = \frac{e^{v_i}}{\sum_{j} e^{v_j}}
$$

In prose: each element $v_i$ is transformed into its exponential, divided by the sum of all the exponentials.

The exponential function $e^x$ is very sensitive to its inputs, and in particular, large numbers cause it to change a lot. This means that if two numbers are 10% different, the exponential function and thus softmax will react a lot more if they're bigger. For example, here are three vectors that each have two numbers 10% from each other:

$$
\begin{align}
\sigma( & [1.0, 1.1] ) &= [0.475, 0.525] \\
\sigma( & [10, 11] ) &= [0.269, 0.731] \\
\sigma( & [100, 110] ) &= [0.000045, 0.999955]
\end{align}
$$

The top of this page has a download link to an interactive softmax visualizer, if you want to play around with it.

[Softmax]: https://en.wikipedia.org/wiki/Softmax_function
:::

To keep softmax from becoming too extreme, we first scale the attention scores by $\sqrt{\delta}$. This factor comes from statistics. Remember that the dot product is the sum of $\delta$ terms, one per dimension. These terms are roughly independent, so the standard deviation of their sum grows as $\sqrt{\delta}$ (this is standard statistics, which we don't need to get into the details of here). By dividing by $\sqrt{\delta}$, we keep the typical magnitude of attention scores consistent regardless of the embedding dimension. This ensures that softmax operates in a reasonable range where it can learn nuanced attention patterns.

Basically, as $\delta$ grows, so do the dot products' variance. This growth happens by a factor of $\sqrt{\delta}$, and left unchecked it would cause softmax to lose nuance between values that are actually fairly close. Dividing by $\sqrt{\delta}$ lets softmax keep that nuance.

This "scaling plus softmax" is called, appropriately enough, the scaled dot-product attention. When it's applied to the raw attention scores we calculated earlier, the result is the normalized {dfn}`attention weights`.

### Attention weights and $W_v$ 🡒 context vector

:::{aside}
{drawio}`attention weights combine with values to form the context vector|images/05/llm-flow-self-attention-context`
:::

All of the work until now has been to calculate the attention weights, which are a $n$-sized vector of scalars that answer the first component of attention: "for each input A, how much does it care about input B?" Now we'll answer the second component: what _is_ B, in the context of our self-attention layer?

We'll start with familiar ground, by turning our $d$-sized input embeddings into $\delta$-sized vectors by multiplying them by a weight matrix. This time we'll use the $W_v$ weight, and the result is a {dfn}`value vector`. As with the key vector, we have one such value vector per input token.

From here, we calculate intermediate "weighted values" by multiplying each value vector by its corresponding attention weight. For example, let's say:

- Attention weight $4$ is 0.27. This means that 27% of the query token's attention is devoted to input 4.
- Input token 4's value vector is $[6.2, 1.4, 7.9]$ (here we have $\delta = 3$).

In this case, the weighted value vector for input 4 is:

$$
\begin{align}
  & 0.27 \cdot [6.2, 1.4, 7.9] \\
= & [(0.27 \cdot 6.2), (0.27 \cdot 1.4), (0.27 \cdot 7.9)] \\
= & [1.67, 0.378, 2.13]
\end{align}
$$

We do this calculation for each of the input tokens to get all of the weighted value vectors. Then, we simply sum those up to get the {dfn}`context vector`. (To sum up vectors, we just sum up their corresponding elements.)

### Recap

Recall that all of this happened from the perspective of a single input, which we called the query token. For this token, we:

1. Calculated a query vector ($\text{query token} \cdot W_q$)
2. Dot-producted that query vector against every input's key vector ($\text{token} \cdot W_k$) to get, for each input, a raw attention score
3. Normalized those attention scores into probabilities, one per input
4. Applied each of those probabilities to a corresponding value vector ($\text{token} \cdot W_v$) to get weighted value vectors, one per input
5. Summed up those weighted value vectors to get a single context vector

All of this gives us the $\delta$-dimensional context vector for that one query token. We then repeat this for each of the $n$ inputs, and the result is our attention layer's output: the full {dfn}`attention output matrix`, or just {dfn}`attention output` for short. This has one context vector for each input, so it's an $n \times \delta$ matrix.

{drawio}`attention weights combine with values to form the context vector|images/05/llm-flow-self-attention-output-matrix`

## Real-world improvements

The above covers the fundamental aspects of how self-attention works, but there are several crucial ways that it's augmented in real-world LLMs. None of these are very complicated, so the hardest part is behind us. Still, it's important to know about these if you want to understand how real LLMs work.

### Dropout

:::{important} TODO
for overfitting
:::

### Causal masking

:::{important} TODO
so we train on what we'll actually be using it for
:::

### Multi-head

:::{important}
each head learns a different aspect
:::

### Output projection

:::{important}
TODO $W_o$
:::

### RoPE

:::{important}
TODO

- [RoPE (Medium)](https://medium.com/@mlshark/rope-a-detailed-guide-to-rotary-position-embedding-in-modern-llms-fde71785f152)
:::

### Multi-layer

:::{important}
TODO -- very quick mention, since we haven't actually touched on NN fits in or how the whole LLM fits together
:::


## "The context is full"

- You may have encountered this. Now you can know what it means.
- Two main drivers:
  - attention matrix includes the $n \times n$ attention scores and weights. This means attention scales as the square of input length.
    - this happens per head per layer, so it really adds up
    - (note that the Q/K/V matrices are each $n \times \delta$, so they scale linearly)
  - In RoPE, positional encoding is trained as part of attention as well; so if we include inputs longer than what was trained on, the model won't have the data to accurately predict

## Mathematical optimizations

:::{important} TODO
brief discussion of implementation: the fact that the conceptual matrices get lifted into tensors with an extra dimension, for batching
:::
