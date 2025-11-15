# Self-attention

## What and _why_ is self-attention?

In [the previous section](./02-input-to-vectors), I described how to turn input text into a list of vectors. In the next section, we'll be feeding those vectors into a neural network. But first, in this section, we're going to use a process called {dfn}`self-attention` to determine how much each word in the input affects the other words in the input.

{drawio}`Self-attention sits between tokenization and the neural net|images/05/llm-flow-self-attention`

This piece of the model is the crucial innovation that GPT-style LLMs introduced over previous ML models. It lets the LLM encode not just what any one word means, but what each word means in the context of the _specific_ words that precede before it. It's not enough to know what "have" means, or even what it means as the third word of a sentence: we want to know that it means something different in "we'll always {u}`have`" as compared to "Houston, we {u}`have`".

Remember:
- The input to self-attention is the input embeddings we computed previously. These are are a vector of vectors: one input embedding per token in the input, and each input embedding is a vector (whose size is a fixed dimensionality that the LLM designer decides).

To do that, we're going to build a matrix that describes, for every token in the input, how much that token cares about every other word. (The LLM will actually compare tokens, but using words as the example makes it easier to talk about.) In LLM lingo, we ask how much each word {dfn}`attends` to each other word, and we call these the {dfn}`attention weights`.

{drawio}`5 by 5 attention weight grid with "Houston we have a problem" as both rows and columns. Each cell shows how much the row word attends to the column word.|images/05/attention-weights-houston`

In contrast to the other [fundamental concepts](#conceptual-layers) in LLMs, attention weights really are a matrix, not a vector.

The attention weights themselves are [activations](#parameter-vs-activation) based on the specific prompt that the LLM is looking at. But, each one is derived not just from the corresponding inputs, but also to learned parameters, as I'll be explaining in the next section.

## Computing attention weights

:::{important} TODO
overview

- weights -> scores -> scaling and softmax -> attention weights
- input embeddings + matrices -> vectors. Why boil down to vectors?
:::

### Query, Key, and Value Weights ($W_q$, $W_k$, $W_v$)

:::{important} TODO
:::

### Calculating a single attention score

:::{important} TODO
:::

### Scaling & Softmax

:::{important} TODO
:::

### Applying scores to get attention weights

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
