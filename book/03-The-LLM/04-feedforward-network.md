# Feedforward network

## Overview

In the self-attention layer, we took input embedding vectors and translated them into context vectors that described what each token meant in relation to the other tokens in the input. Now, we'll pass those context vectors through something called a {dfn}`feedforward network`, which will draw higher-level inferences about those tokens that we'll use to ultimately predict the next token.

{drawio}`The feedforward network is the last step of the LLM|images/ffn/llm-flow-ffn`

## What is a feedforward network (FFN)?

At a high level, a {dfn}`feedforward network (FFN)` takes an input vector, transforms it through a series of learned vector parameters, and spits out an output vector. In that sense, it's similar to some of the transformations we saw in the previous chapter on self-attention. But FFNs add one more twist: the ability to apply transformations _conditionally_. This lets different parts of the FFN specialize on particular patterns in the input.

:::{important} An FFN by any other name
Until now, I've been spelling out "feedforward network" in this book, because it's been an unfamiliar and thus jargony word. But from here on, I'll be referring to it as an FFN.
:::

:::{seealso} What does "feedforward" mean?
:class: dropdown

Feedforward networks are just one corner within the broader field of machine learning (ML). There are lots of disciplines within ML, but the ones relevant to our taxonomy are:

- {dfn}`Neural networks`: Architectures that model data as weighted connections between nodes. (These architectures are inspired by biological neurons --- like the ones in our brains.)
  - {dfn}`Feedforward networks (FFNs)`: Neural networks in which information only flows in one direction (that is, doesn't produce loops).
    - {dfn}`Multi-layer perceptrons (MLPs)`: FFNs in which the network is organized into layers, each of which is fully connected to the ones before and after it.

GPT-style LLMs use MLPs, but the standard literature refers to them by the more general term "FFN". I'll be keeping that convention.
:::

An FFN consists of multiple layers: an input, an output, and one or more {dfn}`hidden layers` consisting of {dfn}`neurons` (sometimes called {dfn}`nodes`). Between each layer are learned parameters that transform one layer to the next.

If our input layer has dimension $d_{in}$ and we want an output with $n$ dimensions, we'll accomplish this by creating $n$ neurons. Each neuron consists of two sets of learned parameters:

- a vector of size $d_{in}$
- a scalar, which we call a {dfn}`bias`

For each neuron, we'll:

1. Take the dot product of the input and learned vector; this gives us a scalar.
2. Add the bias.
3. Pass that sum through an {dfn}`activation function`, which I'll explain in just a moment, and which produces another scalar

This gives us one value per neuron, which is its activation. Since we have $n$ neurons, these activations are our output vector.

(ffn-overview-diagram)=
{drawio}`Inputs feed into neurons, each of which produces one value of the output vector|images/ffn/overview`

:::{aside}

- **number of neurons**: hyperparameter; determines output dimension
- **neuron weights**: learned parameters; one per neuron, and each is a vector of size $d_{in}$
- **biases**: learned parameters; one per neuron, and each is a scalar
- **activation function**: hyperparameter
- **neuron activations**: activations (unsurprisingly!), one per neuron; these form the layer output
:::

:::{warning} Confusing terminology
"Neuron" and "layers" are somewhat ambiguous terms that conflates the learned parameters, the computations that involve them, and the resulting activations. I'll try to be clear about which I mean as we go.

The parameters feeding into a layer (in the true sense of the activation) are sometimes called the {dfn}`layer parameters`.
:::

Each of these neurons essentially defines a pattern the FFN can detect. For example, you may have one neuron that specializes in looking for happy words, another that looks for angry words, and another that looks for something unrelated to sentiment, like past tense. (We'll get into more detail later about how these specializations emerge via training. If you need a refresher of the intuitive version, you can reread [the training analogy](#training-analogy) from the earlier overview chapter.)

### Bias parameters

We need the {dfn}`bias` because each of these neurons defines a linear function in the input's $d_{in}$-dimensional space. The bias lets us compute those functions even if they don't pass through the origin:

{drawio}`diagram showing liner regression intersecting the y axis at about 2.4|images/ffn/bias`

(activation-function)=

### Activation function

Finally, we define the {dfn}`activation function`. This can technically be any non-linear function that takes the raw output from the linear function ( $(input \cdot weights) + bias$ ) and produces another scalar. In practice, a common one is the Rectified Linear Unit (ReLU) function, which is a fancy name for "negative values are truncated at 0":

$$
ReLU(x) = \max(0, x)
$$

The activation function is crucial for neuron specialization, because it lets each neuron deactivate when the input is sufficiently misaligned with the pattern that the neuron detects. This has two main benefits:

- It lets the neuron signal that it hasn't detected what it's looking for.
- It treats all such highly-misaligned values as equivalent, which means that at training time, it won't learn from them. (This is good, because if a neuron is looking for happy words, we don't want it to learn anything from "purple"!)

Combined, these two benefits get at the real power of ReLU, and of FFNs in general: they combine linear functions in a non-linear way, and thus let us find complex, non-linear patterns in the input.

:::{warning} More confusing terminology

This chapter has talked about two different concepts with similar names:

- The {dfn}`activation function` is a hyperparameter that's the same for every neuron in a given layer; it's basically just a line of code in the model.
- The {dfn}`activations` are scalars that are computed at inference (and training), and are derived from the specific inputs (as well as the layer's learned parameters).

In addition, throughout this book we've been using "activations" to refer to _any_ value that's derived from inputs during inference. The activations in this chapter are the origin of this term: the other activations are called that essentially as a metaphor to the ones in this chapter.

The term "activation" comes from the biological metaphor that I mentioned above was the inspiration for neural networks. Just as biological neurons fire in a living being in response to specific stimuli, so do the neurons in our FFN, thanks to the activation function.
:::

### Multiple layers

In a generic FFN, we would have some arbitrary number of hidden layers. Each hidden layer's output is the next layer's input, until the last one produces the FFN's overall output. These layers can produce a hierarchy of increasingly complex concepts: one may identify features like happy words or active voice; another may recognize patterns that combine happy words with active voice verbs; another may detect a pattern that builds off of this happy-plus-active pattern; and so on. Each of these hidden layers, as well as the final output layer, may have any number of neurons.

(multiple-layers-figure)=
{drawio}`A FFN with two hidden layers|images/ffn/multi`

In LLMs, we typically only have one hidden layer per FFN, so the simplified model I described above is actually the full story. (LLMs have a slightly different approach to achieving the sophistication that a multi-layered FFN would provide, as I'll discuss more in @05-putting-it-together).

## Fitting the FFN into the LLM

:::{warning} TODO
Let's do! From Claude:

You'll probably want to cover:

- Position-wise application (FFN processes each token position independently)
- The typical transformer FFN structure (two layers: expand then contract)
- How it fits in the transformer block (after attention, before next block)
- Dimensionality: typically d_model → 4*d_model → d_model
:::
