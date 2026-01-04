# Feedforward network

## Overview

In the self-attention layer, we took input embedding vectors and translated them into context vectors that described what each token meant in relation to the other tokens in the input. Now, we'll pass those context vectors through something called a {dfn}`feedforward network`, which will draw additional inferences.

:::{drawio} images/ffn/llm-flow-ffn
:alt: The feedforward network is the last step of the LLM
:::

## What is a feedforward network (FFN)?

At a high level, a {dfn}`feedforward network (FFN)` takes an input vector, transforms it through learned vector parameters, and spits out an output vector. In that sense, it's similar to some of the transformations we saw in the previous chapter on self-attention. But FFNs add one more twist: they contain groups of learned parameters, called neurons, that activate selectively based on the input. Each neuron can specialize on a different pattern, which makes FFNs great for learning isolated facts.

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

An FFN consists of multiple layers: an input, an output, and one or more {dfn}`hidden layers` between them. Each layer consists of {dfn}`neurons` (sometimes called {dfn}`nodes`). Between each layer are learned parameters that transform one layer into the next.

## Components of an FFN layer

If one layer has dimension $d_{in}$ and the next layer has $d_{out}$ dimensions, we'll define $d_{out}$ neurons. Each neuron transforms the input vector into one output scalar, called the neuron's {dfn}`activation`; this basically defines how aligned the input is to the pattern that neuron is looking for.

:::{drawio} images/ffn/overview-high-level
:alt: input layer connected to neurons connected to output layer
:::

To do this, each neuron defines two sets of learned parameters:

- a weight vector of size $d_{in}$
- a scalar, which we call a {dfn}`bias`

For each neuron, we'll:

1. Take the dot product of the input and the neuron's weight vector; this gives us a scalar.
2. Add the bias.
3. Pass that sum through an {dfn}`activation function`, which I'll explain in just a moment, to produce the neuron's activation.

This gives us one value per neuron, which is its activation. Since we have $d_{out}$ neurons, these activations are the layer's output vector.

(ffn-overview-diagram)=
:::{drawio} images/ffn/overview
:alt: Inputs feed into neurons, each of which produces one value of the output vector
:::

:::{aside}

- **number of neurons**: hyperparameter; determines output dimension
- **neuron weights**: learned parameters; one per neuron, and each is a vector of size $d_{in}$
- **biases**: learned parameters; one per neuron, and each is a scalar
- **activation function**: hyperparameter
- **neuron activations**: activations (unsurprisingly!), one per neuron; these form the layer output
:::

:::{warning} Confusing terminology
"Neuron" and "layers" are somewhat ambiguous terms that often conflate the learned parameters, the computations that involve them, and the resulting activations. I'll try to be clear about which I mean as we go.

The parameters that are used to compute a layer's activations are sometimes called the {dfn}`layer parameters`.
:::

Each of these neurons essentially learns a pattern in the input. For example, you may have one neuron that specializes in looking for happy words, another that looks for angry words, and another that looks for something unrelated to sentiment, like past tense. (We'll get into more detail later about how these specializations emerge via training. If you need a refresher of the intuitive version, you can reread [the training analogy](#training-analogy) from the earlier overview chapter.)

### Weight vector and bias parameters

Each neuron's weight vector and bias define a linear function in the input's $d_{in}$-dimensional space:

$$
\text{linear output} =
\underbrace{(w_1 \cdot input_1) + (w_2 \cdot input_2) + \dots}_{\text{dot product of weight vector and input}} + b
$$

Note that this is _not_ defining a best-fit linear regression on the input data. A better mental model is that the weights define a direction in $d_{in}$-space, and the bias defines a minimum divergence from that direction before the neuron fires, as we'll see in the next section.

To see what I mean by divergence from the direction, let's take just one of the terms:

$$
(w_k \cdot input_k)
$$

If the learned weight parameter $w_k$ and the actual input $input_k$ have the same sign (both positive or both negative), this term will be positive, and the input is aligned with the neuron on this dimension. If the weight parameter and input have opposite signs, they're misaligned.

- If $w_k$ is large, the alignment or misalignment is amplified; this is an important component of the weight vector's direction.
- If $input_k$ is large, the alignment or misaligned is also amplified; this is an important component in the input.

(activation-function)=

### Activation function

The activation function is an FFN's special sauce. This can technically be any non-linear function that translates a scalar to another scalar, but to be useful, the activation function needs a couple other properties. We don't need to get into those properties yet, though they'll come up when I discuss training later (TODO: make sure I do this).

A common activation function is the Rectified Linear Unit (ReLU) function, which is a fancy name for "negative values are truncated to 0":

$$
ReLU(x) = \max(0, x)
$$

:::{drawio} images/ffn/relu
:alt: graph of ReLU
:::

This activation function is where the bias comes in: the higher the bias is, the easier it is for any given input to survive the ReLU cutoff. This means that the higher the bias, the more lax the neuron is about what it considers relevant input. (Of course, the bias can also be negative, meaning the neuron is even stricter than the weights alone would be.)

The activation function is crucial for neuron specialization because it lets each neuron deactivate when the input is sufficiently misaligned with the pattern that the neuron detects (for ReLU, this means when the neuron's activation is negative). This has two main benefits:

- It lets the neuron say that it hasn't detected what it's looking for.
- It treats all such highly-misaligned values as equivalent, which means that at training time, it won't learn from them. (For example, if a neuron is looking for happy words, we don't want it to learn anything from "purple"!)

Combined, these two benefits get at the real power of FFNs: they let each neuron effectively ignore inputs that don't pertain to the pattern it's learning, which lets the FFN as a whole learn many different patterns.

:::{warning} More confusing terminology

This chapter has talked about two different concepts with similar names:

- The {dfn}`activation function` is a hyperparameter that's the same for every neuron in a given layer; it's basically just a line of code in the model.
- The {dfn}`activations` are the neuron's scalar values that are computed at inference (and training).

In addition, throughout this book, I've been using "activations" to refer to _any_ value that's computed from inputs (as opposed to learned parameters). The activations in this chapter are the origin of this term: the other activations are called that essentially as a metaphor to the ones in this chapter.

The term "activation" comes from the biological metaphor that I mentioned above was the inspiration for neural networks. Just as biological neurons fire in a living being in response to specific stimuli, so do the neurons in our FFN, thanks to the activation function.
:::

## Multiple layers

An FFN can have any number of hidden layers. Each hidden layer's output is the next layer's input, until the last one produces the FFN's overall output. These layers can produce a hierarchy of increasingly complex concepts: one may identify features like happy words or active voice; another may recognize patterns that combine happy words with active voice verbs; another may detect a pattern that builds off of this happy-plus-active pattern; and so on.

(multiple-layers-figure)=
:::{drawio} images/ffn/multi
:alt: A FFN with two hidden layers
:::

In LLMs, we typically only have one hidden layer per FFN, as we'll see in the next chapter. (LLMs have a slightly different approach to achieving the sophistication that a multi-layered FFN would provide.)

## Fitting the FFN into the LLM

:::{warning} TODO
Let's do! From Claude:

You'll probably want to cover:

- Position-wise application (FFN processes each token position independently)
:::
