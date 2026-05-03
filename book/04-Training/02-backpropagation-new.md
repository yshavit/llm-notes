---
math:
  '\pdv': '\frac{\partial #1}{\partial #2}'
---

# Backpropagation (v.2)

:::{status} 0
:::

## Introduction

:::{warning}
TODO write intro: 

- > Backpropagation, informally known as "backprop", ...
- > Everyone says it's "an efficient application of the chain rule", but what does that actually mean?
:::

To get an understanding of how backprop works, we'll start exceedingly simple and build up from there:

1. Backprop on a single-layer, scalar model
2. Backprop on a multi-layer, but still scalar model
3. Backprop on a single-layer, matrix-based model
4. Backprop on a multi-layer, matrix-based model

By the last of those, we'll have a "full" understanding of backprop. After that, the only difference between what we've built and a real model is that the real model is bigger.

## The math you'll need

This chapter assumes you're decently familiar with derivatives; if you're not, it may be tough. If you're familiar with them but just need a quick refresher, the following sections should help.

### Derivatives

If we have some function $y = f(x)$, then its derivative $y' = f'(x)$ is how fast $y$ grows at any given point $x$.

We can also express the derivative using what's called Leibniz notation: $\frac{dy}{dx}$. This notation makes it explicit that we're differentiating with respect to $x$.

### Chain rule

The chain rule lets you deconstruct a function that's the composition of two functions --- in other words, a function that takes the output of one function and passes it to another:

$$
h(x) = z( \; y(x) \; )
$$

To compute $h$'s derivative, we:

- take $y$'s derivative at $x$: $y'(x)$
- take $z$'s derivative at $y(x)$: $z'(y(x))$
- multiply them:
  $$
  h'(x) = z'(y(x)) \, y'(x)
  $$

$h'$ is the derivative of $z$ with respect to $x$, so an alternate notation for that (which makes the "with respect to" explicit) is:

$$
h' \longleftrightarrow \frac{dz}{dx}
$$

Using that, we can write the chain rule in a fraction-like way using Leibniz notation:

$$
\def\t#1{\textit{\scriptsize #1}}
\def\tt#1#2{\begin{array}{c}\t{#1}\\\t{#2}\end{array}}
\begin{array}{ccccc}
\frac{dz}{dx}                  & =      & \frac{dz}{dy}                 & \cdot      & \frac{dy}{dx} \\
\tt{``The derivative}{of z wrt x} & \t{is} & \tt{the derivative}{of z wrt y} & \t{times} & \tt{the derivative}{of y wrt x.''}
\end{array}
$$

Note that this isn't actually a fraction; Leibniz notation just illustrates (by way of analogy) how the $dy$ elements "cancel out".

We can intuit why the chain rule works by going back to $h$'s definition, of $z$ evaluated at $y(x)$. So, to see how fast $h$ increases as $x$ increases, we take how fast $y(x)$ increases as $x$ increases, and multiply it by how fast $z$ increases as $y(x)$ increases.

### Partial derivatives

In the above sections, $y$ was defined in terms of a single variable, $x$. But what if there are two variables, or more?

$$
y = f(x, u, v, \dots)
$$

To handle this, we use {dfn}`partial derivatives`. The concept is simple: treat all but one of the variables as a constant, and then take the (ordinary) derivative with respect to that one remaining variable. The Leibniz notation for this is $\pdv{y}{x}$ if $x$ is the "with-respect-to" variable. We can define as many partial derivatives as there are variables:

$$
\pdv{y}{x} \quad,\quad \pdv{y}{u} \quad,\quad \pdv{y}{v} \quad,\quad \dots
$$

## Backprop on a simple, scalar model

To start our intuition for how backprop works, let's start with the simplest possible model, using all scalars:

$$
y = ax + b 
$$

This is just a plain old line, like you learned about in middle school. We're going to use machine learning to figure out its slope and $y$-intercept. Our training data will be a bunch of $(x, y)$ pairs that we'll assume are arranged in more or less a line:

:::{div}
:class: hidden dark:block
![plot of data showing points more or less along a line](/images/backprop/xy-plot-dark.svg)
:::

:::{div}
:class: dark:hidden
![plot of data showing points more or less along a line](/images/backprop/xy-plot-light.svg)
:::

Since we're assuming (as the model designers) that the points form a line $y = ax + b$, our job will be to figure out $a$ and $b$ from the various $(x, y)$ data points.

Our first step is to define a {dfn}`loss function` $L$, which defines how wrong a given prediction is from the true value. A common one is mean squared error (MSE):

$$
L = (y_{pred} - y_{true})^2
$$

Our simple LLM will:

1. Take an $(x, y_{true})$ pair from the training data set
2. Run $x$ through the model (with whatever $a$ and $b$ we currently have) to produce a prediction, $y_{pred}$
3. Calculate the loss $L = (y_{pred} - y_{true})^2$
4. Apply the chain rule on the two partial derivatives, $\pdv{y}{a}$ and $\pdv{y}{b}$ to calculate the {dfn}`gradients` for $a$ and $b$ (I'll explain this in just a second)
5. Use the gradients to nudge $a$ and $b$ towards where they should be

The gradients for each learned parameter ($a$ and $b$) represent the partial derivative of the loss function with respect to that parameter. In other words, it represents just the mechanical, mathematical question of "as that parameter grows, how fast does the loss grow?" Of course, we want the loss to _shrink_, since it represents how wrong the prediction was. So, we just nudge the parameter in the opposite direction of the gradient.

:::{drawio} images/backprop/training-pipeline
:alt: Visual representation of the steps described above
:::
