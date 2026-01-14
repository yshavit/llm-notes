# Putting it all together

## Overview

So far, we've turned text into tokens, tokens into input embeddings, and augmented the input embeddings into attention. We also went over the basics of FFNs. Now we're ready to put the pieces together. We're almost there!

:::{drawio} images/transformer/llm-flow-transformer
:alt: Self-attention and the FFN combine to create a transformer
:::

There are a few steps to go through. Rather than spelling them out from the start, I'll build them up bit by bit. That means this chapter will have a bit of "just one more thing," but hopefully the trade-off is that it'll provide some of the why as well as the what.

## The little LLM that couldn't

Let's start with the smallest thing that has the basic shape of an LLM. Most of it is what we've already covered:

(smallest-llm-figure)=
:::{drawio} images/transformer/smallest-llm
:alt: Simplified LLM
:::

(Note that in these examples, we'll be treating words as tokens. As discussed in @02-input-to-vectors, the actual tokens are substrings and include punctuation.) Most of this should be familiar by now, but the $W_{out}$ and output logits are new.

The logits (a portmanteau of "logistic unit") are a vector of vectors. The "outer" vector's elements represent token predictions, with each element containing the prediction --- logit --- for the corresponding input token.

:::{drawio} images/transformer/smallest-llm-logits
:alt: Input tokens to output. "The quick brown" translates to "quick brown fox"
:::

:::{aside}

- **logit**: (Activations) The final output of the LLM iteration; a $v$-sized vector. Can also be the whole $n \times v$ output.
:::

Each "inner" vector, or logit, has one scalar per token in the LLM's vocabulary. The values within these logits represent how likely that token is to be the correct prediction .

:::{drawio} images/transformer/smallest-llm-logit-values
:alt: A single logit,
:::

To predict the input's next token, we just need to look at the last logit --- that is, the one that makes a prediction for the last input token --- and pick the token with the highest value. That's the one we'll append to the input and loop back again.

:::{drawio} images/transformer/smallest-llm-logits-zoom
:alt: The LLM takes the highest-value token in the last logit
:::

So, that's our desired output shape: $n$ vectors of size $v$ (where $v$ is the vocabulary size). If you remember, the [output from the FFN](#ffn-output-shape) was a $n$ vectors of size $d$ (the FFN output dimension). This means we need to transform each $d$-vectors into a $v$-vector. Hopefully this transformation is familiar enough by now that you can guess how we do it: we need a $d \times v$ matrix, which I'll call $W_{out}$:

$$
\underbrace{\text{FFN Output}}_{n \times d}
\cdot \underbrace{W_{out}}_{d \times v}
= \underbrace{\text{Logits}}_{n \times v}
$$

This matrix doesn't have a standard name. It can be called the output projection, the LM (language modeling) head, or the unembedding layer. It's a learned parameter matrix.

:::{aside}

- **output projection**: a learned matrix, $d \times v$
:::

Note that unembedding is basically the reverse of the original translation from tokens to token embeddings that we did back in @02-input-to-vectors. Some models even use the same weights in both, to cut down on the model size.

:::{tip} Why do we output $n$ logits?
:class: dropdown

When I learned that the LLM outputs an $n$ logits but that we only use the last one, I found myself wondering why we don't just output a single vector.

Part of the issue is that there isn't actually a great way to turn an $n \times d$ matrix into a $v$ vector (or, equivalently, a $1 \times v$ matrix).

But beyond that, the throwaway logits aren't actually throwaway. They're not used at inference, but at training, they let you check $n$ predictions in one pass. In the example above, our third logit predicted that the token after "brown" is "fox", which we know from our input is the correct prediction. If it had predicted "hen", our training would use this to adjust the model's learned parameters.
:::

At this point, you should understand everything in [the simple LLM diagram above](#smallest-llm-figure). That's all we need in order for the equations and dimensions to all add up, so we could call that an LLM. In practice, such an LLM wouldn't work well: its predictions won't be good enough. So, let's beef it up.

## Stacking transformer blocks

In the previous chapter, I mentioned that a traditional FFN [has multiple hidden layers](#multiple-layers-figure), but that LLMs don't. Instead, LLMs stack multiple transformer blocks.

Since each transformer block is just an attention layer and an FFN with a single hidden layer, to me this feels similar to a traditional, multi-layered FFN, but with an attention layer sitting between each traditional FFN layer. That said, that's not how standard literature describes it. People in the field think of transformers as a different architecture, not as a modification of FFNs.

:::{tip} Attention vs FFN within a transformer
:class: dropdown
You may be wondering why each transformer contains both an attention layer and an FFN. Why won't just one of these suffice?

The short answer is that this architecture just seems to work.

But beyond that, each layer focuses on different kinds of patterns. The attention layer tends to focus on relationships, and the FFN tends to focus on individual facts.

For example, if we were predicting the next token in "The capital of Massachusetts is", then the attention layer is (roughly speaking) responsible for learning that the next word should be whatever the capital of Massachusetts is; and the FFN is responsible for knowing that the capital of Massachusetts is Boston.

This division of labor isn't always so clear-cut in practice, though, and it's a point of active research.
:::

:::{drawio} images/transformer/multi-llm
:alt: Same architecture as the minimal LLM, but with multiple transformer blocks
:::

(llm-stacking-depth)=
A small LLM may have a couple dozen transformer blocks. Large, commercial LLMs will have 80 - 100 or more.

In a typical LLM, within each transformer block, the FFN takes an input of dimension $d$, expands it to a hidden layer with dimension $4d$, and then contracts it back to $d$. This approach (and specifically the $4\times$ dimension multiplier) was mostly just found to empirically work; I don't think it has any deep, _a priori_ rationale.

:::{drawio} images/transformer/ffn-4d
:alt: FFNs within an LLM typically expand an input from size d to a hidden layer of size 4d, and then back down to an output of size d
:::

## Refining deep transformer stacks

At this point, we've stubbed out a basic LLM and built it out with multiple transformer blocks, with each deeper transformer learning different patterns. Generally speaking, the deeper the transformer (that is, the further it is from the input and closer to output), the more sophisticated and complex its patterns.

Unfortunately, each transformer also represents a discontinuity in the flow from input to inference. For example, remember that the output from each attention layer doesn't include everything in its input embeddings: it only outputs the information relevant to the relationships its heads detect. What if a deeper transformer would have needed something that isn't part of that projection?

What we really need is for each of these transformers to augment, rather than fully replace, its input. We'll do this with two different tweaks to the transformer stack: normalization and residual connections.

I'll describe each of these in isolation first --- what they do, how they work --- and then show how both plug into the LLM's architecture.

### Normalization

Normalization is mostly for the benefit of training, which I haven't described yet. For now, the important thing to know is that training relies on gradients, so the more stable those gradients are, the easier it'll be for the training process.

So, the goal of normalization is to ensure that activations (that is, the values that are derived from any particular input) don't vary too wildly. To do this, we'll center the values roughly around 0, and also "squash" them so they're roughly ±1.

To calculate the layer's normalized values:

1. First, we'll calculate the activations' mean and variance.

    :::{note} Refresher on variance
    :class: dropdown
    Variance is a standard measure in statistics that describes how spread out a set of values is.

    We calculate it by computing the values' average; then, computing how far each value is from the average; then squaring those distances; then taking the average of those squares.
    :::

2. Next, we'll get the "plain" normalized values:
   $$
   \frac{\text{activations} - \text{mean}}{\sqrt{\text{variance} + \varepsilon}}
   $$

   (where $\varepsilon$ is some small value, like $10^{-5}$)

   - $(\text{activations} - \text{mean})$ centers the values around 0:

     :::{drawio} images/transformer/values-around-zero
     :alt: subtracting the activations' mean from themselves centers them around zero
     :::

   - Since the variance comes from the square of each value's distance from the mean, $\sqrt{\text{variance}}$ gets us back to the scale of the original values (this is the standard deviation). Dividing by this value normalizes the values to roughly ±1:

     :::{drawio} images/transformer/values-plusminus-1
     :alt: dividing by square root of variance gets all values to be ±1
     :::

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

These parameters basically let the training adjust the normalization: instead of the values being roughly 0±1, they'll be roughly _shift_ ± _scale_. Although the training can technically settle on any values for these parameters, in practice they often stay pretty close to 1 and 0.

Putting all of the above together, we get:

(normalization-function)=

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

Although the [non-linear stacking](#activation-function) of multiple transformer blocks lets the model learn sophisticated patterns, that same non-linearity serves to partially "disconnect" adjacent layers. I described above how attention replaces its input embeddings with only the information relevant for the attention's relationships; FFNs do a similar thing with their inferences. Effectively, attention and FFNs both act as a hurdle between their inputs and outputs.

This has two main problems:

- At inference (which flows from input to output) layers may discard information that deeper layers would have needed
- At training (which flows backwards, from inference to input), the hurdle makes it harder for the feedback signals to travel up the stack.

The solution is simple: just add each layer's original activation values back to the post-transformed values. This is called a {dfn}`residual connection`.

:::{aside}

- **residual connection**: also called "skip connection" or "shortcut connection"; a stateless operation (no learned or hyperparameters)
:::

:::{drawio} images/transformer/residual-connection
:alt: A residual connection just adds the pre-transformed activations to the post-transformed activations
:::

The result combines the original values with the result of the transformation. This provides just enough of a connection to let the training flow back up the stack: each layer adjusts its input rather than replacing it entirely.

Residual connections don't add any new learned parameters (or hyperparameters) to the model. They're just a stateless operation between layers.

### Where they fit in

Now that we have normalization and residual connections, we just need to fit them into the overall architecture. We will add:

- a normalization layer before each attention layer and each FFN
- a residual connection around each attention layer and FFN
- a final normalization layer before the final output projection

Each transformer looks like:

:::{drawio} images/transformer/transformer-with-residuals
:alt: Transformer showing normalization and residual connections
:::

I mentioned back in the chapter on attention that the attention layer's output dimension $\delta$ is usually the same as its input $d$. Residuals are one of the main reasons we set that constraint. Without it, we'd need to add yet another transformation to match the dimensions before performing the addition. While this could technically work, it works against the residual's main goal, which is to provide a direct path of data flow throughout the LLM's layers. It's better to just set the two dimensions as equal, so that we can straightforwardly add the residual.

:::{note} Pre- vs post-normalization
:class: dropdown
This approach is called "pre-normalization", since the normalization is before each sub-layer (attention and FFN) in the transformer blocks. When the normalization is [specifically LayerNorm](#normalization-schemes), this is also called "Pre-LN".

Some models also use post-normalization, but this is less common. There isn't a conceptual difference: pre-normalization has just been empirically found to work better.
:::

The overall architecture looks like:

:::{drawio} images/transformer/full-architecture
:alt: Full architecture, with all transformers, normalizations, and residual connections
:::

## Control flow and special tokens

The last thing we need to add to our LLM is the concept of special tokens.

To motivate this, consider the autocomplete loop I've been referencing at the top of every chapter so far: the LLM starts with some input tokens, predicts the next token, appends that to the tokens list, and then starts the loop again with that newly expanded list. But how does it know when to stop?

Until now, I've been using words to illustrate tokens, like "Houston" or "jumps". I mentioned earlier that [LLMs actually tokenize on subcomponents of words](#typical-tokenization), but LLMs also introduce special tokens that describe the high-level structure of the text.

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

## Celebration time!

:::{aside}
:class: big

🎉

&nbsp;&nbsp;&nbsp;&nbsp;🎉
:::

At this point, we've covered all the major components of inference! Some of the concepts I've introduced are a bit outdated (especially the positional embeddings in @02-input-to-vectors), but the newer approaches are refinements, not fundamental or structural changes to the architecture.

Pause for a moment! This is a nice milestone! You now basically understand how LLM inference works.

The next chapter will just describe some algebraic reformulations we can apply to make this translate better to optimized hardware, and then I'll go into training.
