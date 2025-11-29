# Turning input text into vectors

## Overview

As I've mentioned before, vectors are how LLMs encode the nuance of human language. So, the first thing we need to do is to turn each part of the text input into a vector. In the end, we'll have one vector per token in the input text.

{drawio}`Self-attention sits between tokenization and the feedforward network|images/input/llm-flow-input`

We're going to process the input in three steps:

1. Tokenizing the input
2. Looking up {dfn}`embedding vectors` for each token in the input
3. Using that to generate the {dfn}`input embedding` vectors for the input

Each of these steps is very simple, so if you read the following and think "I must be missing something", you probably aren't.

## Tokenization

We start with the input text, which we parse into tokens. This doesn't involve any AI: it's basically just "a" → {keyboard}`<1>`, "aardvark" → {keyboard}`<2>`, etc.

(typical-tokenization)=
I actually won't cover the tokenization algorithm itself, because it's not really an ML/AI topic (it was originally invented for compression!). Suffice it to say that the most common form of tokenization is byte pair encoding (BPE), which basically looks for words, common sub-words like the "de" in "demystify", and punctuation. You can read more about it [on Wikipedia][bpe] if you want. OpenAI has a page that lets you see how text tokenizes: <https://platform.openai.com/tokenizer>.

But basically, this is just turning something like "to be or not to be" into a stream of tokens: {keyboard}`to` {keyboard}`be` {keyboard}`or` {keyboard}`not` {keyboard}`to` {keyboard}`be`.

## Token embeddings

All of the tokens our model knows about form its vocabulary, and each one is associated with a vector called the {dfn}`token embedding`. This embedding's values are learned parameters that encapsulate what the LLM "knows" about that token. The size of each vector is a hyperparameter denoted $d$.

:::{aside}

- {dfn}`token embedding dimension`: integer denoted $d$; hyperparameter
- {dfn}`token embedding`: vector of size $d$; learned parameter
:::

Every token has exactly one embedding that's used throughout the model. If the token appears multiple times in the input, each one will use the same token embedding. (There'll be other things, in particular the @03-self-attention described in the next chapter, to differentiate between input tokens.)

:::{note} Reminder of what these values mean
As mentioned in {ref}`the training analogy <training-analogy>`, these values are just values that emerge through training. If we intuitively think of the various aspects of the word "be" — that it can be a semantically light auxiliary verb, that it can denote existence, that it's used in philosophical existentialism, and so on — then each of these is, very roughly by way of an analogy, a value in the token embedding vector. For example, index 1318 in the embedding vector for "be" may encode its existential connotation. In the embedding token for "dog", index 1318 embedding vector may connote fluffiness instead of existentialism.

Again it's important to remember that the values don't _actually_ encode existentialism or fluffiness. They're just values which settle into being during training, and which correlate with predictive power when generating words.
:::

## Adding positions to get to input embeddings

So, now we have a bunch of token embeddings — one per token in our vocabulary — and we've applied them towards the parsed text:

{drawio}`images/input/token-embeddings`

But a word may mean something different if it's the first word of a sentence vs if it's at the middle or end. That could be because it has an entirely different meaning, and the different usages correlate with position; or it could have the same meaning, but with different nuance or tone. To capture this additional information, we're going to add {dfn}`positional embedding`.

:::{important}
Modern LLMs don't actually use positional embeddings anymore. They still care about positions, but the mechanism is different and more complex. I'll discuss positional embeddings now because they're simpler, and {ref}`beyond-the-toy-llm` will explain the modern alternative.
:::

Just as we defined a unique embedding for each token in the vocabulary — "be" always the same token embedding, for example — we'll now define a unique embedding for each position. For example, the first token in an input always used the same embedding, that of position 0. These embeddings are learned vectors, with the same dimension $d$ as the token embeddings.

:::{aside}

- **positional embedding**: vector of size $d$; learned parameter
:::

Then, for each token in the parsed text, we just the sum its token embedding and positional embedding to get its {dfn}`input embedding`:

{drawio}`images/input/token-and-positional-embeddings`

(Note that I picked the token and positional embedding values so that it'd be easier to follow them through the flow. In an actual LLM, these would all be just random-looking numbers.)

Now that we have the input tokenized, and each token translated into an input embedding, we'll look at how the LLM contextualizes these embeddings relative to each other.

[bpe]: https://en.wikipedia.org/wiki/Byte-pair_encoding
