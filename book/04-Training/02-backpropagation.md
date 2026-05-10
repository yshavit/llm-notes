---
math:
  '\pdv': '\tfrac{\partial #1}{\partial #2}'
  '\dpdv': '\frac{\partial #1}{\partial #2}'
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

This chapter assumes you're decently familiar with derivatives; if you're not, it may be tough. If you're familiar with them but just need a quick refresher, the following sections should help. If you're already comfortable with these, feel free to [jump ahead](#backdrop-single-layer-scalar) to the meat of it.

### Derivatives

If we have some function $y = f(x)$, then its derivative $y' = f'(x)$ is how fast $y$ changes at any given point $x$.

We can also express the derivative using what's called Leibniz notation: $\frac{dy}{dx}$. This notation makes it explicit that we're differentiating with respect to $x$.

To differentiate a polynomial, bring each exponent down as a factor and lower it by one:

$$
\begin{array}{rccccl}
y  & = & ax^n          & +          & bx^m          & + \, \dots \\[0.3em]
   &   & \downarrow    &            & \downarrow    &            \\[0.3em]
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

(backdrop-single-layer-scalar)=
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
\dpdv{L}{a}
$$

We can think of $L$ as a composed function $L(x) = ( \, y(x) \, - y_{true} )^2$. That means we can use the chain rule:

$$
\dpdv{L}{a} = \dpdv{L}{y} \cdot \dpdv{y}{a}
$$

Let's start by calculating the right term, $\pdv{y}{a}$:

$$
y(x) = ax + b \\[0.3em]
\downarrow \\[0.3em]
\pdv{y}{a} = x
$$

Now the left term, $\pdv{L}{y}$:

$$
L(x) = (y(x) - y_{true})^2\\[0.3em]
\downarrow \\[0.3em]
\pdv{L}{y} = 2(y(x) - y_{true})
$$

Putting it all together:

$$
\begin{align}
\dpdv{L}{a} & = \dpdv{L}{y} & \cdot & \; \dpdv{y}{a} \\[1em]
& = 2(y(x) - y_{true}) & \cdot & \; x
\end{align}
$$

And here's where the "efficient application of" starts to kick in: during our inference phase, we already calculated $y(x) = y_{pred}$. If we just store that value during that forward pass, $\pdv{L}{a}$ becomes a trivial calculation: $y(x)$ comes from that stored lookup, and $x$ and $y_{true}$ were our given arguments. We call this value $a$'s {dfn}`gradient`.

We can can do the same thing to calculate $\pdv{L}{b}$. I'll go a bit faster, since it's basically the same work.

$$
\begin{array}{rccc}
\pdv{L}{b} = & \underbrace{\pdv{L}{y}} & \cdot & \underbrace{\pdv{y}{b}} \\[1em]
& \textit{\footnotesize same as $\pdv{L}{y}$ above} & & \footnotesize \pdv{}{b} (ax + b)\\[1.5em]
= & 2(y(x) - y_{true}) & \cdot  & 1
\end{array}
$$

Notice that the left term is exactly the same as it was for $a$'s gradient.

With that, we've calculated our two gradients, for $a$ and $b$. Now we just apply each one to its respective parameter ($a$ and $b$) to update them. As I mentioned before, we _subtract_ the gradients, because we want to _reduce_ the loss. Before we do that, we scale the gradients down by $\eta$, which is a {dfn}`learning rate`. This is some small number, like 0.01, and it means that each round of learning only _nudges_ the values towards a 0-loss, instead of lurching them there. This prevents over-fitting any one data point, which can cause the model to overshoot and oscillate around the desired value, or worse, shoot off to infinity.

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

## Terminology: residual and local derivative

Before we go further, let's introduce two useful names for the concepts we've already learned. Remember that the gradients for $a$ and $b$ each used the chain rule, and in both cases their left-hand term was the same:

$$
\begin{align}
\pdv{L}{a} & = 2(y(x) - y_{true}) & \cdot & \; x \\[1em]
\pdv{L}{b} & = 2(y(x) - y_{true}) & \cdot & \; 1
\end{align}
$$

Let's ask where these various terms come from, and do so within the framing of the layer that contains $a$ and $b$ (that is, $y = ax + b$).

- $2(y(x) - y_{true})$ comes purely from the layer below us ($ \, L(v) = (v - y_{true})^2 \, $), where $y_{true}$ can be thought of as a constant)
  - $v$ got stored during forward inference
  - The fact that we need to $2 \times$ the value is due to the derivative _of $L$_ --- irrespective of what anything else in the model is doing.
- The $x$ and $1$ each come from partial derivatives local to the $y$ layer. Again, these only depend on the $y$ layer, irrespective of what anything else is doing.

The distinction between information coming from the layer below, and information computed at this layer, is reflected in terminology:

- The {dfn}`residual` is the left-hand term in the chain rule: the signal from the layer below
- The {dfn}`local derivatives` are the right-hand term in the chain rule: the partial derivatives applied at this layer

We can think of this for any parameter $p$ as:

(residuals)=
$$
\begin{array}{lll}
\dpdv{L}{p} & = \text{(signal from lower level)} & \cdot \; \text{(partial derivative of $p$)} \\[0.5em]
            & = \text{(residual)} & \cdot \; \text{(local derivative)} \\[1em]
            & = \boxed{r \cdot \dpdv{y}{p}} &
\end{array}
$$

...where:

- $p$ is a parameter defined at layer $y$
- $r$ is the residual, which comes from the layer below $y$

Note that $r$ isn't an equation, but an actual, concrete value. Each layer gets this value, and then uses it as-is for _all_ of that layer's parameters. This is a lot of what's behind the "efficient" in "efficient application of the chain rule".

The lowest layer, $L$, is a special case: it doesn't have a lower layer to provide a residual, so we need to calculate it by figuring out its derivative and plugging in $y_{pred}$ and $y_{true}$.

## Backprop on a multi-layer, scalar model


Now that we have backprop working on a single-layer model, let's add a second layer. For now, we won't have an activation function between the two:

$$
y_1 = a_1x + b_1 \\
y_2 = a_2 (y_1) + b_2
$$

We'll use the same loss function as before:

$$
L(x) = (y_2(x) - y_{true})^2
$$

Let's start by keeping in mind our objectives:

- We want to figure out how much to nudge $a_1$, $b_1$, $a_2$, and $b_2$.
- To do that, we need to calculate their four gradients.
- Each gradient is a partial derivative: $\pdv{L}{a_1}$, $\pdv{L}{b_1}$, $\pdv{L}{a_2}$, $\pdv{L}{b_2}$.

We'll start at the bottom of the model, the layer closest to $L$: $y_2$. This means we'll be calculating the gradients for $a_2$ and $b_2$, which are $\pdv{L}{a_2}$ and $\pdv{L}{b_2}$. Let's start with $a_2$. As before, we'll use the chain rule:

$$
L(x) = (y_2(x) - y_{true})^2 \\[0.3em]
\downarrow \\[0.3em]
\pdv{L}{a_2} = \pdv{L}{y_2} \cdot \pdv{y_2}{a_2}
$$

This turns out to be exactly the same as the single-layer example above: just add a $_2$ subscript to $y$, $a$, and $b$:

$$
\pdv{L}{a_2} = 2(y_2(x) - y_{true}) \cdot x \\[0.3em]
\pdv{L}{b_2} = 2(y_2(x) - y_{true}) \cdot 1
$$

So far, this is all just a review of the previous two sections. Now comes the new wrinkle: calculating the gradients for the $y_1$ layer.

There are two ways to approach this: by working everything out piece by piece, or by relying on the residual-based pattern we established in the previous section. I'm not sure which is more helpful, so I'll provide both. If one doesn't make sense, try the other!

:::::{dropdown} Working it out piece by piece
:open:

As before, we'll start by writing out the loss function:

$$
L(x) = (y_2(x) - y_{true})^2
$$

Note that even though we're interested in the parameters at layer $y_1$, the loss function is still defined in terms of $y_2$. The loss function can _only_ be defined against $y_2$, because its semantic is "how wrong was the model's ultimate prediction"; we don't have any way of estimating how far off an intermediate value was, because our training data only has the inputs and final expected outputs.

Let's start with $\pdv{L}{a_1}$. Again, we'll use the chain rule --- but what do we want to use as the chain?

$$
\dpdv{L}{a_1} = \dpdv{L}{\boxed{\scriptstyle ?}} \cdot \dpdv{\boxed{\scriptstyle ?}}{a_1}
$$

If we think about what $a_1$ most directly impacts --- that is, what changes most directly as $a_1$ changes --- it's just $y_1$, the function that directly uses $a_1$. So, let's use that:

$$
\dpdv{L}{a_1} = \dpdv{L}{y_1} \cdot \dpdv{y_1}{a_1}
$$

With that in mind, let's take a crack at the left term: $\pdv{L}{y_1}$. We can't just use a plain polynomial derivative formula as we've been doing so far, because $y_1$ isn't "directly" in the definition for $L$. Instead, let's try the chain rule again:

$$
\dpdv{L}{y_1} = \dpdv{L}{\boxed{\scriptstyle ?}} \cdot \dpdv{\boxed{\scriptstyle ?}}{y_1}
$$

Again we ask, what does $y_1$ most directly affect? Well, it's used in the next layer:

$$
y_2 = a_2 (\underline{y_1}) + b_2
$$

...so $y_1$ most directly affects $y_2$. Let's fill that in:

$$
\dpdv{L}{y_1} = \dpdv{L}{y_2} \cdot \dpdv{y_2}{y_1}
$$

Here's where the "efficient application" kicks in again. We already computed $\pdv{L}{y_2}$ in the previous step --- it was the left-hand term of the chain rule --- so we can just plug that in. For the right hand term (of our $y_1$ layer), we can fall back to standard derivatives:

$$
\dpdv{y_2}{y_1} \, = \dpdv{}{y_1} (a_2 (y_1) + b_2) \, = a_2
$$

With that, we've calculated the left term of our $a_1$ gradient:

$$
\begin{array}{lccccc}
\dpdv{L}{a_1} & = & \dpdv{L}{y_1} & \cdot & \dpdv{y_1}{a_1} \\[2em]
& = & \pdv{L}{y_2} \cdot \pdv{y_2}{y_1} & \cdot & \pdv{y_1}{a_1} \\[2em]
& = & \underbrace{ \left( \pdv{L}{y_2} \cdot a_2 \right) }_{\text{from previous layer}} & \cdot & \pdv{y_1}{a_1}
\end{array}
$$

And we can then fill in the right term straightforwardly:

$$
\begin{align}
\pdv{L}{a_1} & = \left( \pdv{L}{y_2} \cdot a_2 \right) \cdot & \underbrace{\pdv{y_1}{a_1}}_{\pdv{}{a_1}a_1 x + b_1} \\[2em]
& = \left( \pdv{L}{y_2} \cdot a_2 \right) \cdot & x
\end{align}
$$

The gradient for $b_1$ would work the same way. In the end, you'd get:

$$
\begin{align}
\pdv{L}{b_1} & = \left( \pdv{L}{y_2} \cdot a_2 \right) \cdot & \underbrace{\pdv{y_1}{b_1}}_{\pdv{}{b_1}a_1 x + b_1} \\[2em]
& = \left( \pdv{L}{y_2} \cdot a_2 \right) \cdot & 1
\end{align}
$$

:::::

:::::{dropdown} Using residuals
:open:

If we trust our understanding of the residuals pattern in the previous section:

:::{embed} #residuals
:::

...then we can get a shortcut for all of the above. Let's give a name for the residual coming into $y_1$: $r_1$.

$$
r_1 = \dpdv{L}{y_1}
$$

The $y_2$ layer will calculate this for us, as an extra step after it calculates its gradients. That's because calculating $\pdv{L}{y_1}$ requires knowing $y_1$'s definition. The calculation is cheap:

$$
\begin{array}{lll}
r_ 1 &=& \dpdv{L}{y_1} \\[0.5em]
& & \downarrow \textit{chain rule} \\[0.5em]
&=& \dpdv{L}{y_2} \cdot \dpdv{y_2}{y_1} \\[0.5em]
&=& r_2 \cdot \text{(local derivative of $y_1$ wrt $y_2$)}
\end{array}
$$

The $y_2$ layer knows its definition in terms of $y_1$, so it's able to determine its $\pdv{}{y_1}$:

$$
y_2 = a_2( \, y_1 \, ) + b_2 \\[0.3em]
\downarrow \\[0.3em]
\textit{rewrite to make $y_1$ more explicitly the variable} \\[0.3em]
\downarrow \\[0.3em]
y_2 = y_1 \, a_2 + b_2 \\[0.3em]
\downarrow \\[0.3em]
\dpdv{y_2}{y_1} = a_2
$$

Following this through, we get a pattern for every layer $y_n$:

1. Take the residual from the previous layer. Use it as the left-hand term of chain rule applications, with the right hand term being the local derivative of each parameter $p$ in this layer.
2. Calculate the local derivative of the input ( $y_{n-1}$ ), and pass to the $y_{n-1}$ layer

:::::

## Adding an activation function

TODO

## Using tensors instead of scalars

TODO
