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

Within each transformer block, the FFN's hidden layer takes input of dimension $d$, expands it to dimension $4d$, and then contracts it back to $d$. This approach was mostly just found to empirically work; I don't think it has any deep, _a priori_ rationale.

(llm-stacking-depth)=
A small LLM may have a couple dozen transformer blocks, and large, commercial ones have 80-100 or more.

## Architectural tweaks to aid training

If all we had to worry about were inference, we'd be done at this point. Unfortunately, we still need to train our model, and deeply stacked transformers are going to cause issues when we do so. To solve this, we're going to add two new ideas: normalization and residual connections.

:::{note} Getting ahead of ourselves, by necessity
Normalization and residual connections are all about training, which I haven't talked about yet.

These components aren't part of the LLM's core conceptual architecture in the same way that attention or even FFNs are. They're "just" engineering workarounds that have been empirically found to make training better. Training is a crucial part of creating a good LLM, so these pieces are extremely important in practice; but I'll cover training later, so don't worry if the motivation for adding them doesn't click yet.
:::

Without getting into technical details, it turns out that our stacked transformers would have two problems at training time:

1. Activations that jump too wildly from layer to layer will make it hard for the model to learn stable patterns.
2. The deep transformer stacking acts as a dampening effect, such that layers earlier in the model (and thus farther from the prediction that training will check against) will receive a much softer training signal than later layers.

We'll solve the first problem with {dfn}`normalization`, and the second with {dfn}`residual connections`.

I'll describe what each layer is first, and then show how they fit into the architecture.

### Normalization

The goal of normalization is to ensure that activations (that is, the values at inference time) don't vary too wildly. Our goal is to center the values roughly around 0, and also "squash" them so they're roughly ±1.

To calculate the layer's normalized values:

1. First, we'll calculate the activations' mean and variance.

    :::{note} Refresher on variance
    :class: dropdown
    Variance is a standard measure in statistics that describes how spread out a set of values is.

    We calculate it by computing the values average; then, computing how far each value is from the average; then squaring those distances; then taking the average.
    :::

2. Next, we'll get the "plain" normalized values:
   $$
   \frac{\text{activations} - \text{mean}}{\sqrt{\text{variance} + \varepsilon}}
   $$

   (where $\varepsilon$ is some small value, like $10^{-5}$)

   - $(\text{activations} - \text{mean})$ centers the values around 0:

     {drawio}`subtracting the activations' mean from themselves centers them around zero|images/transformer/values-around-zero`

   - Since the variance comes from the square of each value's distance from the mean, $\sqrt{\text{variance}}$ gets us back to the scale of the original values (this is the standard deviation). Dividing by this value normalizes the values to roughly ±1:

     {drawio}`dividing by square root of variance gets all values to be ±1|images/transformer/values-plusminus-1`

   - Adding $\varepsilon$ basically provides a minimum value for the denominator to avoid division by 0.
     :::{seealso} Details on $\varepsilon$
     :class: dropdown
     Variance is always ≥ 0 (since it comes from squares of values), but if it's exactly 0 then we'll get a division-by-zero error, and even if it's just extremely small (like $10^{-15}$ or something), the result of the division would be huge. This usually represents a rounding error rather than the true value we want, so adding $\varepsilon$ counteracts that.

     Adding it unconditionally, as opposed to only when the values are small enough to require it, is more efficient at the hardware level.
     :::

3. Finally, we'll multiply by a learned parameter called {dfn}`scale` and add a learned parameter called {dfn}`shift`.

The scale and shift are both vectors of size $d$, the dimension of the layer to be normalized. This just lets us scale and shift each activation dimension separately. In contrast to most learned parameters, the scale and shift are _not_ totally random: they start as 1 and 0, respectively.

:::{aside}

- **normalization scale**: a learned $d$-sized vector
- **normalization shift**: a learned $d$-sized vector
:::

These parameters basically let the training adjust the normalization: instead of the values being roughly 0±1, they'll be roughly _shift_ ± _scale_.

Although the training can technically settle on any values for these parameters, in practice they often stay pretty close to 1 and 0.

Putting all of the above together, we get:

$$
\text{scale}
\cdot \left(
  \frac{\text{activations} - \text{mean}}{\sqrt{\text{variance} + \varepsilon}}
\right) + \text{shift}
$$

:::{note} Other normalization schemes
(normalization-schemes)=
The above algorithm is called LayerNorm. There are other algorithms, some of which are more sophisticated and some which are simpler (and thus cheaper to compute). They all serve the same high-level function, so I won't go into them in detail.
:::

### Residual connections

Although the [non-linear stacking](#activation-function) of multiple transformer blocks lets the model learn sophisticated patterns, that same non-linearity serves to "disconnect" adjacent layers. At training, this can dampen the effect that each lower layer has on the one above it. Since LLMs typically have [many layers](#llm-stacking-depth), this presents a real problem in practice.

The solution is simple: just add each layer's original activation values back to the post-transformed values. This is called a {dfn}`residual connection`.

:::{aside}

- **residual connection**: also called "skip connection" or "shortcut connection"; a stateless operation (no learned or hyperparameters)
:::

{drawio}`A residual connection just adds the pre-transformed activations to the post-transformed activations|images/transformer/residual-connection`

The result combines the original values with the result of the transformation. This provides just enough of a connection to let the training flow back up the stack: each layer adjusts its input rather than replacing it entirely.

Residual connections don't add any new learned parameters (or hyperparameters) to the model. They're just a stateless operation between layers.

### Where they fit in

Now that we have normalization and residual connections, we just need to fit them into the overall architecture. We will add:

- a normalization layer before each attention layer and each FFN
- a residual connection around each attention layer and FFN
- a final normalization layer before the final output projection

Each transformer looks like:

{drawio}`Transformer showing normalization and residual connections|images/transformer/transformer-with-residuals`

:::{note} Pre-normalization
:class: dropdown
This approach is called "pre-normalization", since the normalization is before each sub-layer (attention and FFN) in the transformer blocks. When the normalization is [specifically LayerNorm](#normalization-schemes), this is also called "Pre-LN".

Some models also use post-normalization, but this is less common. There isn't a conceptual difference: pre-normalization has just been empirically found to work better.
:::

The overall architecture looks like:

{drawio}`Full architecture, with all transformers, normalizations, and residual connections|images/transformer/full-architecture`

## Control flow and special tokens

The last thing we need to add to our LLM is the concept of special tokens.

To motivate this, consider the autocomplete loop I've been referencing at the top of every chapter so far: the LLM starts with some input tokens, predicts the next token, appends that to the tokens list, and then starts the loop again with that newly expanded list. But how does it know when to stop?

Until now, I've been using words to illustrate tokens, like "Houston" or "jumps". I mentioned earlier that [LLMs tokenize on subcomponents of words](#typical-tokenization), but LLMs also introduce special tokens that describe the high-level structure of the text.

The most important of these is probably {keyboard}`<EOS>`, or end-of-sequence. This token means that the output text is done, and when an LLM outputs one of these, it knows to stop the loop. (Different models may call this other things, like {keyboard}`<|endoftext|>`.)

For example, in the following dialogue, {keyboard}`<EOS>` signifies that the line is over, and that it's the next person's turn to speak:

> **MERCUTIO** I am hurt. A plague o' both your houses! I am sped. Is he gone, and hath nothing?{keyboard}`<EOS>`
>
> **BENVOLIO** What, art thou hurt?{keyboard}`<EOS>`
>
> **MERCUTIO** Ay, ay, a scratch, a scratch; marry, 'tis enough.
Where is my page? Go, villain, fetch a surgeon.{keyboard}`<EOS>`

Some other special tokens may include {keyboard}`<System>` / {keyboard}`<User>` / {keyboard}`<Assistant>` for defining different roles within a chat interface.

These affect the UX of the LLM, but not its core AI, so I won't go into much depth on them. Just know that they exist, and in particular that {keyboard}`<EOS>` acts as the signal that the LLM should stop the loop and consider its text generation done.

## In summary

:::{aside}
:class: big

🎉

&nbsp;&nbsp;&nbsp;&nbsp;🎉
:::

At this point, we've covered all the major components of inference! Some of the concepts I've introduced are a bit outdated (especially the positional embeddings in @02-input-to-vectors), but the newer approaches are refinements, not fundamental or structural changes to the architecture.

Nice!
