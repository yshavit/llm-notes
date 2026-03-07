---
math:
  '\tok': '\boxed{\texttt{#1}\vphantom{X}}\;'
  '\merge': '\; \underbrace{ \tok{#1} \tok{#2}} \;'
---
# Turning input text into vectors

:::{status} 2
:::

## Overview

As I've mentioned before, vectors are how LLMs encode the nuance of human language. So, the first thing we need to do is to turn each part of the text input into a vector. In the end, we'll have one vector per token in the input text.

:::{drawio} images/input/llm-flow-input
:alt: Self-attention sits between tokenization and the feedforward network
:::

We're going to process the input in three steps:

1. Tokenizing the input
2. Looking up {dfn}`embedding vectors` for each token in the input
3. Using that to generate the {dfn}`input embedding` vectors for the input

Each of these steps is pretty simple, so if you read the following and think "I must be missing something", you probably aren't.

## Tokenization

We start with the input text, which we parse into tokens --- essentially, the atoms of the input text.

(typical-tokenization)=
Tokenization doesn't involve "real" AI: it's basically just "a" → {keyboard}`<1>`, "aardvark" → {keyboard}`<2>`, and so on. The most common form of tokenization is byte-pair encoding (BPE), which basically looks for words, common sub-words like the "de" in "demystify", and punctuation. OpenAI has a page that lets you see how text tokenizes: <https://platform.openai.com/tokenizer>.

BPE isn't a true ML/API topic: it was originally invented for compression. As such, you can feel free to skip the details if you like.

:::::{seealso} BPE details
:class: dropdown

BPE tokenization is relatively simple, at least in its unoptimized form. We start with two configurations:

- a priority-ordered list of merge pairs (for example, $[a \; b] \rightarrow ab\,$)
- a mapping from token to ID (for example, $ab \rightarrow 1432\,$)

Both of these configurations operate on bytes, not ASCII characters or unicode (hence the term _byte_-pair encoding).

These configurations are generated during the LLM's training. As with other training-related parameters, I won't discuss how they're generated; during inference, we just assume they're provided.

At a high level, the BPE steps are:

1. Encode the incoming text as UTF-8
2. Merge byte sequences using the merge list
3. Convert the resulting sequences to IDs using the token mapping

Let's look at each of these. To keep things simple, I'll keep all the characters as ASCII, and represent them by their ASCII letters instead of bytes; so, '$\texttt{a}$' instead of $\texttt{0x61}$. Just remember that this is really operating on bytes, not characters.

1. UTF-8 decoding

   This is just what it sounds like: we encode the text to bytes using UTF-8. We then treat each byte as a 1-byte sequence:

   $$
   \text{"Hi Bob!"} \\ \downarrow \\[ 0.5em ]
   \begin{array}{ccccccc}
   \texttt{H} & \texttt{i} & \texttt{\char"00B7} & \texttt{B} & \texttt{o} & \texttt{b} & \texttt{!} \\
   \texttt{(0x48} & \texttt{0x69} & \texttt{0x20} & \texttt{0x42} & \texttt{0x6f} & \texttt{0x62} & \texttt{0x21)} \\
   \tok{H} & \tok{i} & \tok{\char"00B7} & \tok{B} & \tok{o} & \tok{b} & \tok{!} \\
   \end{array}
   $$

2. Merge sequences:

   At this point, we have a list of byte sequences (so, a list of lists). Each of the inner lists has exactly 1 element, but that's about to change.

   Now, we go through the merge pairs in priority order. For each merge pair, we look for consecutive sequences that match that pair; if we find them, we merge them into a single sequence.

   For example, if the merge list is:

   ```text
   [ B ]   [ o ]
   [ H ]   [ i ]
   [ Bo ]  [ b ]
   ```

   ...then we'll merge:

   $$
   \begin{align}
   & \tok{H} \tok{i} \tok{\char"00B7} \merge{B}{o} \tok{b} \tok{!} & - \; \textit{merge \tok{B}\tok{o}} \\[1em]
   & \merge{H}{i} \tok{\char"00B7} \tok{Bo} \tok{b} \tok{!} & -  \;\textit{merge \tok{H}\tok{i}} \\[1em]
   & \tok{Hi} \tok{\char"00B7} \merge{Bo}{b} \tok{!} & - \;\textit{merge \tok{Bo}\tok{b}} \\[1em]
   & \tok{Hi} \tok{\char"00B7} \tok{Bob} \tok{!} \\
   \end{align}
   $$

3. Finally, we'll use the token mappings to convert each of these sequences to an ID.

   For example, if the token mappings are:

   ```text
   [ Hi ]  → 1
   [ Bob ] → 2 
   [ ! ]   → 3
   [   ]   → 4
   ```

   Then we'll map:

   $$
   \begin{array}{cccc}
   \tok{Hi} & \tok{\char"00B7} & \tok{Bob} & \tok{!} \\[0.5em]
   \downarrow & \downarrow & \downarrow & \downarrow \\[0.5em]
   1 & 4 & 2 & 3
   \end{array}
   $$

The only wrinkle is that as we go through the priority list (in step 2), we may create sequences whose merge pairs we already passed. For example, imagine if the merge pairings above had had a different priority:

```text
[ Bo ] [ b ]
[ B ] [ o ]
[ H ] [ i ]
```

In this case, we'd do:

$$
\begin{align}
& \tok{H} \tok{i} \tok{\char"00B7} \tok{B} \tok{o} \tok{b} \tok{!} & - \;\textit{no \tok{Bo}\tok{b} to merge} \\[1em]
& \tok{H} \tok{i} \tok{\char"00B7} \merge{B}{o} \tok{b} \tok{!} & - \;\textit{merge \tok{B}\tok{o}} \\[1em]
& \merge{H}{i} \tok{\char"00B7} \tok{Bo} \tok{b} \tok{!} & - \;\textit{merge \tok{H}\tok{i}} \\[1em]
& \tok{Hi} \tok{\char"00B7} \tok{Bo} \tok{b} \tok{!} & - \;\textit{never merged \tok{Bo}\tok{o}!} \\
\end{align}
$$

To solve this, once we find a merge pair, we start reset the merge pairs list and look from the top again:

$$
\begin{align}
& \tok{H} \tok{i} \tok{\char"00B7} \tok{B} \tok{o} \tok{b} \tok{!} & - \;\textit{no \tok{Bo}\tok{b} to merge} \\[1em]
& \tok{H} \tok{i} \tok{\char"00B7} \merge{B}{o} \tok{b} \tok{!} & - \;\textit{merge \tok{B}\tok{o}; reset search} \\[1em]
& \tok{H} \tok{i} \tok{\char"00B7} \merge{Bo}{b} \tok{!} & - \;\textit{merge \tok{Bo}\tok{b}} \\[1em]
& \tok{H} \tok{i} \tok{\char"00B7} \tok{Bob} \tok{!} & - \;\textit{no \tok{B}\tok{o} to merge} \\[1em]
& \merge{H}{i} \tok{\char"00B7} \tok{Bob} \tok{!} & - \;\textit{merge \tok{H}\tok{i}} \\[1em]
& \begin{array}{cccc}
\tok{Hi} & \tok{\char"00B7} & \tok{Bob} & \tok{!} \\
\downarrow & \downarrow & \downarrow & \downarrow \\
1 & 4 & 2 & 3
\end{array}
\end{align}
$$

There are optimization tricks we can do to make this more efficient, but in terms of the core logic, that's it!

:::::

## Token embeddings

All of the tokens our model knows about form its vocabulary, and each one is associated with a vector called the {dfn}`token embedding`. This embedding''s used throughout the model. If the token appears multiple times in the input, each one will use the same token embedding. (There'll be other things, in particular the @03-self-attention described in the next chapter, to differentiate between input tokens.)

Since we've already tokenized the input, now we just need to create a vector of vectors: each outer vector corresponds to one token in the input, and the inner vector is that token's embedding:

:::{drawio} images/input/token-embeddings
:::

:::{note} Reminder of what these values mean
As mentioned in {ref}`the training analogy <training-analogy>`, these values are just values that emerge through training. If we intuitively think of the various aspects of the word "be" --- that it can be a semantically light auxiliary verb, that it can denote existence, that it's used in philosophical existentialism, and so on --- then each of these is, very roughly by way of an analogy, a value in the token embedding vector.

Although every embedding vector is technically independent, training will generally cause them to align what each index means. For example, index 1318 may converge towards meaning something like "single-syllable word" across all embeddings in the LLM's vocabulary.

Again it's important to remember that the values don't _actually_ encode existentialism or syllable count. They're just values which settle into being during training, and which correlate with predictive power when generating words.
:::

## Adding positions to get to input embeddings

A word's meaning may change depending on where in a sentence it appears. That could be because it has an entirely different meaning, and the different usages correlate with position; or it could have the same meaning, but with different nuance or tone. To capture this additional information, we're going to add a {dfn}`positional embedding` to each input.

:::{note}
Modern LLMs don't actually use positional embeddings anymore. They still care about positions, but the mechanism is different and more complex. I'll discuss positional embeddings now because they're simpler, and {ref}`beyond-the-toy-llm` will explain the modern alternative.
:::

Just as we defined a unique embedding for each token in the vocabulary --- "be" always the same token embedding, for example --- we'll now define a unique embedding for each position. For example, the first token in an input always used the same embedding, that of position 0. These embeddings are learned vectors, with the same dimension $d$ as the token embeddings.

:::{aside}

- **positional embedding** (learned parameter): vector of size $d$
:::

For each token in the parsed text, we just the sum its token embedding and positional embedding to get its {dfn}`input embedding`:

:::{drawio} images/input/token-and-positional-embeddings
:::

(Note that I picked the token and positional embedding values so that it'd be easier to follow them through the flow. In an actual LLM, these would all be just random-looking numbers.)

Now we have the input tokenized, and each token translated into an input embedding. In the next chapter, I'll show how the LLM contextualizes these embeddings relative to each other.

[bpe]: https://en.wikipedia.org/wiki/Byte-pair_encoding
