---
math:
  '\pdv': '\frac{\partial #1}{\partial #2}'
---

# Backpropagation

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

If we have some function $y = f(x)$, then its derivative $y' = f'(x)$ is how fast $y$ changes at any given point $x$.

We can also express the derivative using what's called Leibniz notation: $\frac{dy}{dx}$. This notation makes it explicit that we're differentiating with respect to $x$.

To differentiate a polynomial, bring each exponent down as a factor and lower it by one:

$$
\begin{array}{rccccl}
y  & = & ax^n          & +          & bx^m          & + \, \dots \\[1em]
   &   &               & \Downarrow &               &            \\[1em]
y' & = & n \; ax^{n-1} & +          & m \; bx^{m-1} & + \, \dots
\end{array}
$$

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

$h'$ is the derivative of $z$ with respect to $x$, so the Leibniz notation for that is:

$$
h' \longleftrightarrow \frac{dz}{dx}
$$

Using that, we can write the chain rule in a fraction-like way:

$$
\def\t#1{\textit{\scriptsize #1}}
\def\tt#1#2{\begin{array}{c}\t{#1}\\\t{#2}\end{array}}
\begin{array}{ccccc}
\frac{dz}{dx}                  & =      & \frac{dz}{dy}                 & \cdot      & \frac{dy}{dx} \\
\tt{``The derivative}{of z wrt x} & \t{is} & \tt{the derivative}{of z wrt y} & \t{times} & \tt{the derivative}{of y wrt x.''}
\end{array}
$$

Note that this isn't actually a fraction; the Leibniz notation just illustrates (by way of analogy) how the $dy$ elements "cancel out".

We can intuit why the chain rule works by going back to $h$'s definition: $z$ evaluated at $y(x)$. So, to see how fast $h$ changes as $x$ changes, we take how fast $y(x)$ changes as $x$ changes, and multiply it by how fast $z$ changes as $y(x)$ changes.

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

Now that we have our math refreshed, let's get to the fun stuff! To start our intuition for how backprop works, let's start with the simplest possible model: a scalar, linear function:

$$
y = ax + b 
$$

This is just a plain old line, like you learned about in middle school. We're going to use machine learning to figure out its slope and $y$-intercept. Our training data will be a bunch of $(x, y)$ pairs:

:::{div}
:class: hidden dark:block
![plot of data showing points more or less along a line](/images/backprop/xy-plot-dark.svg)
:::

:::{div}
:class: dark:hidden
![plot of data showing points more or less along a line](/images/backprop/xy-plot-light.svg)
:::

Since we're assuming (as the model designers) that the points form a line $y = ax + b$, our job will be to figure out $a$ and $b$ from the various $(x, y)$ data points. In other words, $a$ and $b$ are the model's learned parameters.

Our first step is to define a {dfn}`loss function` $L$, which defines how wrong a given prediction is from the true value. A common one is mean squared error (MSE), which we'll adapt for our scalar model:

$$
L(x) = (y(x) - y_{true})^2
$$

:::{note} Terminology
Here and below, I'll use:

- $y(x)$ (or just $y$, in Leibniz notation) for the function that describes the model
- $y_{true}$ for the correct, known output for a given $x$; the training data consists of $(x, y_{true})$ pairs
- $y_{pred}$ for the predicted value: $y_{pred} = y(x)$
:::

Our simple model will:

1. Take an $(x, y_{true})$ pair from the training data
2. Run $x$ through the model (with whatever $a$ and $b$ we currently have) to produce a prediction, $y_{pred}$
3. Calculate the loss $L = (y_{pred} - y_{true})^2$
4. Use the chain rule to compute the two partial derivatives, $\pdv{L}{a}$ and $\pdv{L}{b}$. These give us the {dfn}`gradients` for $a$ and $b$ (I'll explain this in just a second)
5. Use the gradients to nudge $a$ and $b$ towards their true values

The gradients for each learned parameter ($a$ and $b$) represent the partial derivative of the loss function with respect to that parameter. In other words, it represents just the mechanical, mathematical question of "as that parameter grows, how fast does the loss grow?" Of course, we want the loss to _shrink_, since it represents how wrong the prediction was. So, we just nudge the parameter in the opposite direction of the gradient.

:::{drawio} images/backprop/training-pipeline
:alt: Visual representation of the steps described above
:::

The first three steps in the list above are trivial (remember that in this example, "run $x$ through the model" is just $y_{pred} = ax + b$). Let's focus on the fourth step, the chain rule.

We'll focus on $a$ first. What we want is the partial derivative of the loss $L$ with respect to $a$:

$$
\pdv{L}{a}
$$

We can think of $L$ as a composed function $L(x) = ( \, y(x) \, - y_{true} )^2$. That means we can use the chain rule:

$$
\pdv{L}{a} = \pdv{L}{y} \cdot \pdv{y}{a}
$$

Let's start by calculating the right term, $\pdv{y}{a}$:

$$
y(x) = ax + b \\[0.8em]
\Downarrow \\[0.8em]
\pdv{y}{a} = x
$$

Now the left term, $\pdv{L}{y}$:

$$
L(x) = (y(x) - y_{true})^2\\[0.8em]
\Downarrow \\[0.8em]
\pdv{L}{y} = 2(y(x) - y_{true})
$$

Putting it all together:

$$
\begin{align}
\pdv{L}{a} & = \pdv{L}{y} & \cdot & \; \pdv{y}{a} \\[1em]
& = 2(y(x) - y_{true}) & \cdot & \; x
\end{align}
$$

And here's where the "efficient application of" starts to kick in: during our inference phase, we already calculated $y(x) = y_{pred}$. If we just store that value during that forward pass, $\pdv{L}{a}$ becomes a trivial calculation: $y(x)$ comes from that stored lookup, and $x$ and $y_{true}$ were our given arguments. We call this value $a$'s {dfn}`gradient`.

We can can do the same thing to calculate $\pdv{L}{b}$. I'll go a bit faster, since it's basically the same work.

$$
\begin{align}
\pdv{L}{b} &= \underbrace{\pdv{L}{y}}_{\textit{same as $\partial L / \partial a$ above}} \cdot \underbrace{\pdv{y}{b}}_{\partial/\partial b (ax + b) = 1} \\[3em]
&= 2(y(x) - y_{true}) \cdot 1
\end{align}
$$

$$
\begin{array}{rccc}
\pdv{L}{b} = & \underbrace{\pdv{L}{y}} & \cdot & \underbrace{\pdv{y}{b}} \\[1em]
& \textit{\footnotesize same as $\pdv{L}{y}$ above} & & \footnotesize \pdv{}{b} (ax + b) = 1 \\[1.5em]
= & 2(y(x) - y_{true}) & \cdot  & 1
\end{array}
$$

Notice that the left term is exactly the same as it was for $a$'s gradient.

Finally, we just apply the gradients to our learned parameters $a$ and $b$ to update them. As I mentioned before, we subtract the gradients, because we want to lower the loss. Before we do that, we scale them down by $\eta$, which is a {dfn}`learning rate` that's some small number like 0.01. This means that each round of learning only _nudges_ the values towards a 0-loss, instead of lurching them there; this prevents over-fitting any one data point.

$$
\eta = 0.01 \\[1.5em]
a_{updated} = a - (\eta \; a_{gradient}) \\
b_{updated} = b - (\eta \; b_{gradient})
$$

That's all there is to it! If we churn this training through a large enough data set, $a$ and $b$ will eventually converge to the right values.

:::{tip} Try it out!

The following widget lets you see the training in action.

(If you set the learning rate too high, $a$ and $b$ diverge towards infinity or NaN. This is a real phenomenon, and illustrates the importance of the learning rate! In the widget, this renders as a vertical line.)

```{anywidget} ./linear-backprop.mjs
```

:::

## Backprop on a multi-layer, scalar model

TODO
