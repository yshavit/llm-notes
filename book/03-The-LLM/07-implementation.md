# Implementation

:::{status} 0
:::

Now that we've worked through the conceptual workings of the LLM as well as the reformulations to make it efficient, let's see what all of this actually looks like in code.

There's a fully working implementation of an LLM at [{in-img}`../images/implementation/gh.svg`yshavit/llm-notes`://simpllm`][simpllm] (which is in the same repository that hosts this book's source). That repository has a few modules:

- `simpllm-core`: A complete, from-scratch implementation of an LLM. This includes the tokenizer, inference, tensor math, and logit sampling. The only external dependencies are `rand` (for randomization in the logit sampler) and `rayon` (for parallelization).
- `simpllm`: An executable that takes `simpllm-core` and wraps it into a nice TUI.
- `fasterllm`: An executable that replaces the tensor math in `simpllm-core` with a real, production implementation.
- A few other helper modules.

Of these, `simpllm-core` is the most interesting. Let's take a look at some of the highlights.

## Tokenization

If you recall, the three steps for tokenization are:

1. Convert the input text to UTF-8 bytes
2. Merge the bytes using configured merge-pairs
3. Look up the resulting sequences to find their token IDs

Steps 1 and 3 are trivial, so let's take a look at step 2. It's not too bad!

:::{rustref} BPE::MERGE
:::

## Attention

Attention is a bit involved because of KV caching and the various details like scaling, softmax, and combined $QKV$ weights; but even so, it's not too bad:

:::{rustref} Attention
:::

[simpllm]: https://github.com/yshavit/llm-notes/tree/main/simpllm
