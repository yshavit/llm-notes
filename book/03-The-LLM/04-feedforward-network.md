# Feedforward network

:::{status} 2
:::

## Overview

In the self-attention layer, we took input embedding vectors and translated them into context vectors that described the relationship between tokens. Now, we'll pass those context vectors through something called a {dfn}`feedforward network`, which will draw additional inferences.

:::{drawio} images/ffn/llm-flow-ffn
:alt: The feedforward network is the last step of the LLM
:::

At a high level, a {dfn}`feedforward network (FFN)` takes an input vector, transforms it through learned vector parameters, and spits out an output vector. In that sense, it's similar to some of the transformations we saw in the previous chapter on self-attention. But FFNs add one more twist: they group their learned parameters into bundles that each activate selectively, based on the input. Each bundle can specialize on a different pattern, which makes FFNs great for learning isolated facts.

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

An FFN consists of multiple layers: an input, an output, and one or more {dfn}`hidden layers` between them. Each layer consists of {dfn}`neurons` (sometimes called {dfn}`nodes`). Between each layer are learned parameters and logic that transform one layer into the next.

For example, an FFN with four neuron layers would look like:

:::{drawio} images/ffn/overview-high-level
:alt: overview of four neuron layers, each fully connected to its neighbors via a transformation layer
:::

Note that each layer can have any number of neurons. The number of layers in an FFN, as well as the number of neurons in each layer, are hyperparameters of the model.

## Components of an FFN layer

To show how the transformations work, I'll focus on the transformation from just one layer to the next. Every transformation works fundamentally the same way.

If one layer has $d_{in}$ neurons and the next layer has $d_{out}$, we'll define $d_{out}$ neuron-transformations. Each neuron-transformation treats the input as a $d_{in}$-sized vector and transforms it into one output scalar, called the {dfn}`activation`. Since we have $d_{out}$ of these neuron-transformations, we'll have $d_{out}$ activations: these are the resulting layer.

:::{drawio} images/ffn/overview-high-level-layer
:alt: input layer connected to neurons connected to output layer
:::

To do this, each neuron-transformation defines two sets of learned parameters:

- a weight vector of size $d_{in}$
- a scalar, which we call a {dfn}`bias`

For each neuron-transformation, we'll:

1. Take the dot product of the input and the neuron's weight vector; this gives us a scalar.
2. Add the bias.
3. Pass that sum through an {dfn}`activation function`, which I'll explain in just a moment, to produce the neuron's activation.

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

:::{warning} Terminology confusion
The standard literature for FFNs doesn't have a name for the transformations between layers, so it often calls each stack of weights, bias, and activation a "neuron". Effectively, it conflates the true neuron --- the activation scalar --- with the computational unit that generated it.

I've introduced "neuron-transformations" and "weight vectors" to try to reduce this confusion, but be aware that these are not standard terms.
:::

Each neuron essentially learns one pattern in the input. For example, you may have one neuron that specializes in looking for happy words, another that looks for angry words, and another that looks for something unrelated to sentiment, like past tense. (We'll get into more detail later about how these specializations emerge via training. If you need a refresher of the intuitive version, you can reread [the training analogy](#training-analogy) from the earlier overview chapter. As always, remember that the real patterns are more abstract and opaque than "happy" or "past tense".)

### Weight vector and bias parameters

Each neuron's weight vector and bias define a linear function in the input's $d_{in}$-dimensional space:

$$
\text{linear output} =
\underbrace{(w_1 \cdot input_1) + (w_2 \cdot input_2) + \dots}_{\text{dot product of weight vector and input}} + b
$$

Note that this is _not_ defining a best-fit linear regression on the input data. A better mental model is that the weights define a direction in $d_{in}$-space, and the bias defines a minimum alignment to that direction before the neuron activates, as we'll see in the next section.

To see what I mean by alignment to the direction, let's take just one of the terms:

$$
(w_k \cdot input_k)
$$

If the learned weight parameter $w_k$ and the actual input $input_k$ have the same sign (both positive or both negative), this term will be positive, and the input is aligned with the neuron on this dimension. If the weight parameter and input have opposite signs, they're misaligned on this dimension.

The input's overall alignment to the weight vector is a sum of all these terms, plus the bias.

- If **$w_k$** is large, the alignment or misalignment is amplified; this component of the **weight vector** contributes a lot to the overall alignment.
- Similarly, if **$input_k$** is large, the alignment or misalignment is amplified; this component of the **input** contributes a lot to the overall alignment.

(activation-function)=

### Activation function

If all we had were the above linear transformations, we wouldn't need multiple layers: you could just make a single transformation layer that's effectively a concatenation of all the multi-layer ones, and that would be equivalent. But if you did that, you'd lose the ability to create specialized inferences at each layer, which then combine to create (usually) higher-level inferences in deeper layers.

The activation function is what prevents this collapse. This can technically be any non-linear function that translates a scalar to another scalar, but to be useful, the activation function needs a couple other properties. (We don't need to get into those properties yet, though they'll come up when I discuss training later.) (TODO: make sure I do this)

A common activation function is the Rectified Linear Unit (ReLU) function, which is a fancy name for "negative values are clipped to 0":

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

(ffn-output-shape)=

## Position-wise application

In all of the above, the FFN's input is a vector of scalars. But if you remember, the output of the attention layer was a vector of _vectors_, each called an embedding.

So, in the FFN layer, we just independently pass each of these embeddings through the same FFN. In other words, the FFN layer takes $n$ token embeddings, each of size $d_{in}$; and turns them into $n$ outputs, each of size $d_{out}$.

We don't have one FFN per token position, or anything like that: the same exact FFN --- with the same weights and biases --- gets applied to each input embedding.

:::{drawio} images/ffn/position-wise-application
:alt: Each input embedding (vector of vectors of scalars) is independently processed through the FFN to produce an output embedding (a different vector of vector of scalars)
:::

In practice, GPUs and TPUs can run these computations in parallel and very efficiently.

## Next up

As I mentioned above, an FFN can have any number of hidden layers. Each hidden layer's output is the next layer's input, until the last one produces the FFN's overall output. These layers can produce a hierarchy of increasingly complex concepts: one may identify features like happy words or active voice; another may recognize patterns that combine happy words with active voice verbs; another may detect a pattern that builds off of this happy-plus-active pattern; and so on. (Again, the actual patterns it finds are much more abstract than that.)

(multiple-layers-figure)=
:::{drawio} images/ffn/multi
:alt: A FFN with two hidden layers
:::

In LLMs, though, we typically only have one hidden layer per FFN. LLMs still need the complex inference that deep FFNs provide, but they accomplish it in a slightly different way.

In the next chapter, I'll show how the attention layer and FFN combine into a transformer block, and how we can stack multiple of these blocks together to create the full LLM.
