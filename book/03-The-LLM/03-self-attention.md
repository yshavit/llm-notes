# Self-attention

## What and _why_ is self-attention?

In [the previous section](./02-input-to-vectors), I described how to turn input text into a list of vectors. In the next section, we'll be feeding those vectors into a neural network. But first, in this section, we're going to use a process called {dfn}`self-attention` to determine how each word in the input affects the other words in the input.

{drawio}`Self-attention sits between tokenization and the neural net|images/05/llm-flow-self-attention`

This piece of the model is the crucial innovation that GPT-style LLMs introduced over previous ML models. It lets the LLM encode not just what any one word means, but what each word means in the context of the _specific_ words that precede before it. It's not enough to know what "have" means, or even what it means as the third word of a sentence: we want to know that it means something different in "we'll always {u}`have`" as compared to "Houston, we {u}`have`".

{drawio}`5 by 5 attention weight grid with "Houston we have a problem" as both rows and columns. Each cell shows how the row word attends to the column word.|images/05/attention-weights-houston`

I find it helpful to be clear about the input and output to this attention layer:

- The input is a vector representing each token in the input. Each token is itself represented by an input embedding (as described in the previous chapter), so this is an $n \times d$ matrix, where $n$ is the input size and $d$ is the dimensionality of each embedding.
- The output will be a vector of these attentions outputs. Each attention output is a vector, so this will be a $n \times \delta$ matrix.
  :::{important} Two small but important notes
  - $d$ and $\delta$ are technically independent [hyperparameters](#parameter-vs-activation), but in practice, $\delta = d$ in most LLMs. That's because if $\delta$ were smaller than $d$, you'd be losing information; and if it were bigger, you'd be encoding more information than you started with. It'd be like writing a number to 10 decimal places when your ruler only measures precisely to 2.
  - $\delta$ is _not_ a standard name in LLM literature. I'm picking it to show that it's conceptually different from $d$ while highlighting that in practice they're equal.
  :::

To do this, we're going to first build up the basics of an attention output. Then, we'll introduce some important real-world refinements.

## Computing attention outputs

:::{attention} Hang onto your butts!
I found this all very hard to wrap my head around. I've tried to interweave the "what" of the computations with the "why" of why they work as best I can. Be patient with yourself: this is tricky, but it's crucial to understanding how LLMs work.
:::

### What will we be building, and why?

As I mentioned above, what we really want to answer is: "for every input, how does it relate to every other input?" That question has nuance, and as you'll recall from [my overview](#vectors-are-nuance), nuance means vectors.

So, easy: all we need is an $n \times n$ grid of values. We're trying to translate our $d$-dimensional input embeddings to a $delta$-dimensional output. Remember from the [matrix multiplication](#matrix-multiplication-notes) chapter that $A_{x,y} \cdot B_{y,z} = C_{x,z}$. This means each element in the grid has to be a $d \times \delta$ matrix:

{drawio}`n-by-n grid, where each cell is a delta-sized vector|images/05/attention-weights-houston-vectors`

The problem is that this is huge — far too large to reasonably store and train on.

So, instead of asking "how does input A affect input B", we'll approximate that question by asking two simpler ones:

- How _much_ does input A care about input B?
- What does input B mean?

The first of those we'll represent with a simple scalar; the second is a nuanced question, so we'll use a ($\delta$-dimensional) vector.

Note that our two questions involved three usages of tokens:

- How much does **(input A)** care about **(input B)**?
- What does **input B** mean?

Each of these usages has some nuance, so we'll use a learned transformation for them. We'll call these $W_q$, $W_k$, and $W_v$ (you'll see why in just a moment):

- How much does **(input A transformed by $W_q$)** care about **(input B transformed by $W_k$)**?
- What does **(input B transformed by $W_v$)** mean?

Each of these is $\delta$-dimensional vector. Remember from the previous chapter on [matrix multiplication](#matrix-multiplication-notes) that TODO. If we think of our input embedding vector as a $d \times 1$ matrix, and the $\delta$-dimension vectors we want as $1 \times \delta$ matrices, this means $d \times \delta$ matrices will give us the transformations we need:

$$
Input_{(1 \times d)} \cdot W\star_{(d \times \delta)} = Output_{(1 \times \delta)}
$$

The "query / key / value" terminology comes from an analogy to database lookups, though I personally don't find the analogy useful. In case you do, the idea is:

- $W_q$ ("query weights"): What am I looking for?
- $W_k$ ("key weights"): What do I offer as context?
- $W_v$ ("value weights"): What information should I pass along?

:::{tip}
In trying to understand this, I kept struggling to understand why we needed these various transformations. After all, the input embeddings we previously computed already contain a concept of nuance. What more do these transformations add?

Remember that what we ideal transformation is an $n \times n \times d \times \delta$ tensor. This is too big, so we're essentially using a couple tricks to compress it.

- First, we're replacing the inner $d \times \delta$ matrices with a 0-dimensional scalar.
- Then, we're applying this scalar against each of the tokens.

If we used a directly-learned $d \times \delta$ matrix for the scalars, and treated the "each of the tokens" as just their embeddings, we would be compressing the ideal tensor _too_ much. Adding the three weight matrices lets us recover a lot of the ideal tensor's flexibility, at a fraction of the cost.

Essentially, each matrix lets us augment our knowledge about the input embeddings with learnings about attention specifically.
:::

The next sections will describe how these matrices fit together. If the above doesn't make sense, it may be useful to move on for now, and then re-read it once you understand the mechanics of how we use these weight matrices.

### Overview of how the weights fit together

For each token within the input, we'll focus on that input and do a bunch of calculations centered on it. Let's call that token the query token: it's the token for which we want to ask (that is, "query") how it attends to each token in the input.

1. First, we apply each of the weight matrices:
    1. We apply the query weights ($W_q$) to the query token to get its {dfn}`query`: what is this token looking for?
    2. We apply the key weights ($W_k$) to every token to get the {dfn}`keys`: how can we match against each token?
    3. We apply the value weights ($W_v$) to every token to get the {dfn}`values`: what information does each token contain?
2. We combine the query with each key to compute the {dfn}`attention weights`, one per token. These are scalars that tell us how much the query token should care about each other token.
3. We combine the attention weights with each value to compute {dfn}`weighted values`, again one per token. These are vectors.
4. We combine the weighted values to get the {dfn}`context vector`: a weighted blend of information from all tokens, focused on what's relevant to the query token.

{drawio}`visual representation of the overall flow described above|images/self-attention/overview`

Again, all of that work is just for a single query token. We'll repeat it for each token in the input to produce a context vector per token. The final result is a vector of vectors, which we'll represent as a matrix. This matrix is our self-attention layer's output.

:::{important}

- In contrast to most other [fundamental concepts](#conceptual-layers) in LLMs, the attention parameters really are matrices, not vectors.
- There is only _one_ $W_q$, one $W_k$, and one $W_v$. Each matrix gets paired with various tokens as we'll see below, and its training will reflect that. But there isn't a different $W_q$ per query token, for example. $W_q$ essentially encodes what it means to be a query token, independent of any particular embedding; and similarly for $W_k$ and $W_v$.
- The attention weights themselves are [activations](#parameter-vs-activation) based on the specific prompt that the LLM is looking at.
:::

The following subsections will go into each of these steps in more detail.

### $W_q$ 🡒 query vector

This one is easy.

- We start with the query token's input embedding, which is a vector of size $d$
- We have the query weight matrix $W_q$, of size $d \times \delta$
- We just multiply them together: $query = embedding \cdot W_q$
- Out comes and we get a vector of size $\delta$

That's it! We do this just once per query input.

### Query vector and $W_k$ 🡒 attention score

We'll do this step once for every input token; I'll call these tokens the key tokens (though that's not a standard term).

First, we'll calculate the key for each key token. This is a $\delta$-sized vector. Similar to how we computed the query vector, this key vector is just $key\ embedding \times W_k$.

Now we have two $\delta$-sized vectors: the query and the key. We can compute their [dot product](#dot-product-math) to combine them into a scalar.

Geometrically, a dot product represents how aligned two vectors are, assuming the vectors are normalized. Our query and key vectors have _not_ been normalized; but over training, the $W_q$ and $W_k$ matrices' values will converge to capture meaningful relationships.

In other words, the dot product of these two vectors represents how aligned they are; or, in LLM terms, how much the query "makes sense" relative to the key. This tells us how much the query token cares about the input token that corresponds to this key.

We call this dot product the raw {dfn}`attention score` for this key.

### Attention score 🡒 attention weight

:::{important} TODO
Scaling & Softmax
:::

### Attention weight and $W_v$ 🡒 context vector

:::{important} TODO
:::

### Recap

:::{important} TODO
:::

## Real-world improvements

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

### RoPE

:::{important}
TODO
:::

## Mathematical optimizations

:::{important} TODO
brief discussion of implementation: the fact that the conceptual matrices get lifted into tensors with an extra dimension, for batching
:::
