# Implementation

:::{status} 0
:::

Now that we've worked through the conceptual workings of the LLM as well as the reformulations to make it efficient, let's see what all of this actually looks like in code.

There's a fully working implementation of an LLM at <https://github.com/yshavit/llm-notes/tree/main/simpllm> (which is in the same repository that hosts this book's source). Let's take a look at some of the highlights.

## Tokenization

If you recall, the three steps for tokenization are:

1. Convert the input text to UTF-8 bytes
2. Merge the bytes using configured merge-pairs
3. Look up the resulting sequences to find their token IDs

Steps 1 and 3 are trivial, so let's take a look at step 2. It's not too bad!

:::{rustref} BPE::MERGE
:::
