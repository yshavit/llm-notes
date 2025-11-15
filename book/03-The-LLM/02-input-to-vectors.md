# Turning input text into vectors

## Overview

As I've mentioned before, vectors are how LLMs encode the nuance of human language. So, the first thing we need to do is to turn each part of the text input into a vector. In the end, we'll have one vector per token in the input text.

{drawio}`Self-attention sits between tokenization and the neural net|images/04/llm-flow-input`

## Fundamental concept of input embedding

As I mentioned in @conceptual-layers, I like to think about LLMs in terms of the fundamental concepts, and then the mathematical optimizations. I'll do that here.

We're going to process the input in three steps:

1. Tokenizing the input
2. Looking up {dfn}`embedding vectors` for each token in the input
3. Using that to generate the {dfn}`input embedding` vectors for the input

Each of these steps is very simple, so if you read the following and think "I must be missing something", you probably aren't.

### Tokenization

We start with the input text, which we just parse into tokens.

I actually won't cover the tokenization algorithm itself, because it's not really an ML/AI topic. Suffice it to say that the most common form of tokenization is byte pair encoding (BPE), which basically looks for words, sub-words (like the "de" in "demystify"), and punctuation. You can read about it [on Wikipedia][bpe], but since it's not really an AI concept (it was originally invented for compression!), it's not particularly interesting for us.

But basically, this is just turning something like "to be or not to be" into a stream of tokens: {keyboard}`to` {keyboard}`be` {keyboard}`or` {keyboard}`not` {keyboard}`to` {keyboard}`be`.

### Token embeddings

All of the tokens our model knows about form its vocabulary, and each one is associated with a vector called the {dfn}`token embedding`. This embedding's values are learned parameters that encapsulate what the LLM "knows" about that token.

Every token has exactly one embedding that's used throughout the model. If the token appears multiple times in the input, each one will use the same token embedding. (There'll be other things, in particular the @03-self-attention described in the next chapter, to differentiate between input tokens.)

:::{note} Reminder of what these values mean
As mentioned in @what-are-learned-parameters, these values are just values that emerge through training. If we intuitively think of the various aspects of the word "be" — that it can be a semantically light auxiliary verb, that it can denote existence, that it's used in philosophical existentialism, and so on — then each of these is, very roughly by way of an analogy, a value in the token embedding vector. For example, item 318 in the embedding vector for "be" may encode its existential connotation. These values are particular to each token: an item in the same index 318 for "dog"'s embedding vector may connote fluffiness.

Again it's important to remember that the values don't _actually_ encode existentialism or fluffiness. They're just values which settle into being during training, and which correlate with predictive power when generating words.
:::

### Input embeddings

In a modern LLM, the input embeddings are just the token embeddings. Simple!

{drawio}`images/04/token-embeddings`

## Mathematical optimizations

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

## Positional encoding in old-style LLMs

As mentioned above, modern LLMs just use the token embeddings as the input embeddings. Older LLMs, like GPT-2, also used the token's positions within the input text.

In this approach, each position has its own embedding, called appropriately enough the {dfn}`positional embedding`. Just as the token embeddings are global within the LLM — "be" always uses the same token embedding, for example — so was each positional embedding. For example, the first token in an input always used the same embedding, that of position 0.

Then, the input embedding is just the sum of each input's token embedding and its positional embedding:

{drawio}`images/04/token-and-positional-embeddings`

(Note that I picked the token and positional embedding values so that it'd be easier to follow them through the flow. In an actual LLM, these would all be just random-looking numbers.)

Modern GPTs simplify the input embedding phase, and instead make the self-attention phase (the next chapter) more sophisticated.

[bpe]: https://en.wikipedia.org/wiki/Byte-pair_encoding
