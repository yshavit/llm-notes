# Putting it all together

So far, we've turned text into tokens, tokens into input embeddings, and augmented the input embeddings into attention. We also went over the basics of FFNs. Now we're ready to put the pieces together. We're almost there!

{drawio}`Self-attention and the FFN combine to create a transformer|images/transformer/llm-flow-transformer`

There are a few steps to go through. Rather than spelling them out from the start, I'll build them up bit by bit. That means this chapter will have a bit of "just one more thing," but hopefully the trade-off is that it'll provide some of the why as well as the what.

## The little LLM that couldn't

Let's start with the smallest thing that has the basic shape of an LLM. Most of it is what we've already covered:

(smallest-llm-figure)=
{drawio}`Simplified LLM|images/transformer/smallest-llm`

(Note that in these examples, we'll be treating words as tokens. As discussed in @02-input-to-vectors, the actual tokens are substrings and include punctuation.) Most of this should be familiar by now, but the $W_{out}$ and output logits are new.

The logits (a portmanteau of "logistic unit") are a vector of vectors. The "outer" vector's elements represent token predictions, with each element corresponding to one past the corresponding input token.

{drawio}`Input tokens to output. "The quick brown" translates to "quick brown fox"|images/transformer/smallest-llm-logits`

:::{aside}

- **logit**: The final output of the LLM iteration; a $v$-sized vector. Can also be the whole $n \times v$ output.
:::

Each "inner" vector, or logit, has one scalar per token in the LLM's vocabulary. The values within these logits represent how likely that token is to be the right one.

{drawio}`A single logit,|images/transformer/smallest-llm-logit-values`

Since the last logit in the outer vector represents the predictions for the next token after the input, and the highest value in that logit represents which of those tokens is most likely to be right. That's the one we'll append to the input and loop back again.

If you remember, the output from the FFN was a $n$-sized vector (one per token) of $d$-sized vectors (the FFN's inferences; we'll get to $d$'s sizing below). We need to turn each of these $d$-vectors into a $v$-vector, where $v$ is the vocabulary size. Hopefully this will be enough that you can guess how we do this: we need a $d \times v$ matrix, which I'll call $W_{out}$:

$$
\text{FFN Output}_{(n \times d)} \cdot W_{out\ (d \times v)} = \text{Logits}_{(n \times v)}
$$

This matrix doesn't have a standard name. It can be called the output projection, the LM (language modeling) head, or the unembedding layer. It's a learned parameter matrix.

:::{aside}

- **projection layer**: a learned matrix, $d \times v$
:::

Note that unembedding is basically the reverse of the original translation from tokens to token embeddings that we did back in @02-input-to-vectors. Some models even use the same weights in both, to cut down on the model size.

:::{tip} Why do we output $n$ logits?

When I learned that the LLM outputs an $n$ logits but that we only use the last one, I found myself wondering why we don't just output a single vector.

Part of the issue is that there isn't actually a great way to turn an $n \times d$ matrix into a $v$ vector (or, equivalently, a $1 \times v$ matrix).

But beyond that, the throwaway logits aren't actually throwaway. They're not used at inference, but at training, they let you check $n$ predictions in one pass. For example, if the output of "The quick brown fox" had produced an output that predicted "fox" (input token 4, output logit 3) as unlikely, we could use that as feedback during training. The $n$-logit output gives us $n$ such training opportunities per pass.

We can't optimize this away at inference time, because we'd need to do that via a different projection matrix, and we won't have the training to fill in that matrix's values.
:::

At this point, you should understand everything in [the image above](#smallest-llm-figure).

:::{aside}
:class: big

🎉
:::

Pause for a moment. This is a milestone. You now basically understand how LLM inference works. The rest of this chapter is just about real-world refinements to this fundamental model.

## Stacking transformer blocks

In the previous chapter, I mentioned that a traditional FFN [has multiple hidden layers](#multiple-layers-figure), but that LLMs don't. Instead, LLMs stack multiple transformer blocks.

Since each transformer block is just an attention layer and an FFN with a single hidden layer, to me this feels similar to a traditional, multi-layered FFN, but with that special-sauce of attention sprinkled throughout. (That said, that's not how standard literature describes it. People in the field think of transformers as a different architecture, not as a modification of FFNs.)

{drawio}`Same architecture as the minimal LLM, but with multiple transformer blocks|images/transformer/multi-llm`

Note that LLMs typically add one final attention layer between the last transformer and the output projection, as shown in the diagram.

Within each transformer block, the FFN's hidden layer takes input of dimension $d$, expands it to dimension $4d$, and then contracts it back to $d$. This approach was mostly just found to empirically work; I don't think it has any deep, _a priori_ rationale.

A small LLM may have a couple dozen transformer blocks, and large, commercial ones have 80-100 or more.

## Architectural tweaks to aid training

If all we had to worry about were inference, we'd be done at this point. Unfortunately, we still need to train our model, and deeply stacked transformers are going to cause issues when we do. To solve this, we're going to add two new ideas: normalization and residual connections.

:::{note} Getting ahead of ourselves, by necessity
Normalization and residual connections are all about training, which I haven't talked about yet.

Normalization and residual connections aren't part of the LLM's core conceptual architecture in the same way that attention or even FFNs are. They're "just" engineering workarounds that have been empirically found to make training better. Training is a crucial part of creating a good LLM, so these pieces are extremely important in practice; but I'll cover training later, so don't worry if the motivation for them doesn't click yet.
:::

Without getting into technical details, it turns out that our stacked transformers would have two problems at training time:

1. Activations that jump too wildly from layer to layer will make it hard for the model to learn stable patterns.
2. The deep transformer stacking acts as a dampening effect, such that layers earlier in the model (and thus farther from the prediction that training will check against) will receive a much softer training signal than later layers.

We'll solve the first problem with {dfn}`normalization`, and the second with {dfn}`residual connections`.

### Normalization

:::{warning} TODO
:::

### Residual connections

:::{warning} TODO
:::

:::{warning} TODO

In addition to whatever's in the book, make sure I cover:

1. Residual connections / skip connections: The attention output gets added back to the input (x + Attention(x)). This is crucial for training deep networks. **This is why $d = \delta$. Otherwise, you'd need yet another learned projection.
2. Layer normalization: Usually applied before or after attention (or both). This is important for training stability. Make sure I mention that epsilon is just to avoid dividing by zero.

> Why Not Use a Conditional instead of always adding epsilon?
> You're right that a conditional would work functionally, but there are a few reasons we don't do it that way:
>
> 1. Numerical stability (the main reason):
> Even when variance isn't exactly zero, it can be very close to zero (like 1e-10). Dividing by very small numbers causes numerical instability—you get huge values that can overflow or cause NaN (not-a-number) errors downstream.
> Adding epsilon ensures you're always dividing by at least epsilon (typically 1e-5 or 1e-6), which keeps everything numerically stable.
> 2. Differentiability:
> Conditionals create discontinuities in the function. For backpropagation, you need smooth gradients. The conditional variance == 0 ? epsilon : variance has a sharp jump, which is problematic for gradient-based optimization.
> Adding epsilon smoothly blends in, maintaining differentiability everywhere.
> 3. Hardware efficiency (your intuition):
> Yes, conditionals do cause branch divergence on GPUs/TPUs. Different threads taking different paths through a conditional can serialize execution and kill parallelism. Always adding epsilon avoids any branching.
:::

## Special tokens

:::{warning} TODO

- EOS (End of Sequence)
- UNK (Unknown)
- System / User / Assistant to basically toggle "modes" during chat
:::
