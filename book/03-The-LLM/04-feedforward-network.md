# Feedforward network

In the self-attention layer, we took input embedding vectors and translated them into context vectors that described what each token meant in relation to the other tokens in the input. Now, we'll pass those context vectors through something called a {dfn}`feedforward network`, which will draw higher-level inferences about those tokens that we'll use to ultimately predict the next token.

{drawio}`The feedforward network is the last step of the LLM|images/ffn/llm-flow-ffn`

## What is a feedforward network (FFN)?

At a high level, a {dfn}`feedforward network (FFN)` takes an input vector, transforms it through a series of learned vector parameters, and spits out an output vector. In that sense, it's similar to some of the transformations we saw in the previous chapter on self-attention. But FFNs add one more twist: the ability to apply transformations _conditionally_. This lets different parts of the FFN specialize on particular patterns in the input.

:::{info} What does "feedforward" mean?
:class: dropdown

Feedforward networks are just one corner within the broader field of machine learning (ML). There are lots of disciplines within ML, but the ones relevant to our taxonomy are:

- {dfn}`Neural networks`: Architectures that model data as weighted connections between nodes. (These architectures are inspired by biological neurons — like the ones in our brains.)
  - {dfn}`Feedforward networks (FFNs)`: Neural networks in which information only flows in one direction (that is, doesn't produce loops).
    - {dfn}`Multi-layer perceptrons (MLPs)`: FFNs in which the network is organized into layers, each of which is fully connected to the ones before and after it.

GPT-style LLMs use MLPs, but the standard literature refers to them by the more general term "FFN". I'll be keeping that convention.
:::

Until now, I've been spelling out "feedforward network" in this book, because it's been an unfamiliar and thus jargon-y word. But from here on, I'll be referring to it as an FFN.

:::{warning} WIP
TODO


Also include something like:

> :::{note}
> **Activation vs. Activation Function**
>
> Don't confuse these two related terms:
>
> - **Activation**: The numeric value at a neuron (e.g., 3.7 or 0 or -2.1)
> - **Activation function**: The mathematical function that computes that value (e.g., ReLU, sigmoid, tanh)
>
> For example, if a neuron receives input -1.5 and applies the ReLU function (which outputs max(0, x)), the resulting **activation** is 0.
>
> The term "activation" comes from biological neurons that either "fire" or don't, but in artificial neural networks, activations are continuous numeric values, not just on/off states.
> :::
:::

## Fitting the FFN into the LLM

:::{warning} TODO
Let's do!
:::
