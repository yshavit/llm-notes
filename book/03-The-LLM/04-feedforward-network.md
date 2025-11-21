# Feedforward network

In the self-attention layer, we took input embedding vectors and translated them into context vectors that described what each token meant in relation to the other tokens in the input. Now, we'll pass those context vectors through something called a {dfn}`feedforward network`, which will draw higher-level inferences about those tokens and ultimately predict the next token in the input.

{drawio}`The feedforward network is the last step of the LLM|images/ffn/llm-flow-ffn`

## What is a feedforward network (FFN)?

:::{warning} WIP
TODO

Some terminology:

- Neural Network (broadest term)
  - Feed-Forward Network (information flows one direction)
    - MLP (multiple layers, fully connected, non-linear)

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
