---
math:
  '\note': '\color{gray}{\text{\small #1}}'
---

# Backpropagation

:::{status} 0
:::

:::{warning} Raw notes
This is just raw notes for now. Don't expect it to be readable.
:::

## Terminology

- forward pass
- backpropagation

## Chain rule

Given:
$$
y(x) = f(g(x))
$$

...we can compute the derivative of $y$ with respect to $x$ as:

$$
\frac{d y}{d x} = y'(x) = f'(g(x)) \; g'(x)
$$

## Very simple example

We'll simplify the whole LLM into the very bare minimums to demonstrate how a loss function and gradient computation work. Our "pico LLM" will have:

- a scalar input
- two layers, each of which will multiply its input by a scalar
- a scalar output

$$
\begin{array}{llll}
\textbf{input}   \; &      & = x_0 & \\
\text{layer 1} \; & x_1  & = x_0 \times k_1 & \\
\text{layer 2} \; & x_2  & = x_1 \times k_2 & = \textbf{output}
\end{array}
$$

Our goal will be to find $k_1$ and $k_2$ that fit our data.

:::{tip} Bear with me
:class: dropdown

Yes, this model is _extremely_ simple. You may work out that it can even be simplified to just a single layer with one scalar parameter:

$$
\begin{array}{rrcl}
x_2 & = & x_1 & \times k_2 \\
    & = & (x_0 \times k1) & \times k_2 \\
    & = & x_0 \times (k_1 & \times k_2) \\
    & = & x_0 \times k_s
\end{array}
$$

...where $k_s$ is itself just a scalar, and thus a parameter we could discover directly (rather than via its $x_0$, $x_1$ components --- of which there are an infinite number).

For now, just forget about this $k_s$ shortcut, and let's see how we can train our $x_0$ and $x_1$ parameters.

Once we build up the basics intuition of how backpropagation works, we'll apply it to more sophisticated models, building up to the full LLM as described in the first part of this book.
:::

Remember that our ultimate goal is to figure out our two learned parameters, $k_1$ and $k_2$. Intuitively, what we want to figure out is: "for each learned parameter, figure out how much and in which direction to wiggle it, such that the prediction would have more closely matched the actual value from our training data."

Our general strategy will be:

- Define a {dfn}`loss function` $L$. This function takes two arguments: the result of the forward pass, and the expected value. It returns a single scalar that represents how wrong the forward pass was.
- For each parameter $k$ in the model, differentiate $L$ with respect to $k$ as $L'$, and calculate $L'(k)$. In other words, differentiate the loss function with respect to $k$, computed at $k$.
  - Crucially, we won't actually have to figure out $L'$ in general; we only need to solve $L'(k)$, which ends up being a much simpler problem.

First, let's note that we can turn the model into a single expression. Our output is:

$$
x1 \times k2
$$

...and $x_1$ is $x_0 \times k_1$, so we can just substitute that in:

$$
(x_0 \times k_1) \times k2
$$

Now, we'll our loss function $L$. Any differentiable function will do; let's pick squaring:

$$
L(x, \text{expected}) = (\text{model}(x) - \text{expected})^2
$$

In other words:

- Take the input $x$, run it through the model.
- Subtract the expected value (which is in our training data set).
- Square that; the result is the loss

To solve the various partial derivatives mentioned above ($L$ with respect to each $k$), we're going to apply the chain rule against the various {dfn}`partial derivatives` of the loss function. This just means we'll hold all but one variable constant, and then take the derivative with respect to that one constant. We'll do this starting from the bottom, and work our way up.

First, let's calculate $k_2$. We'll do this by holding all other values constant, and differentiating the loss function with respect to $k_2$. Let's write out $L$ again, this time substituting in our model:

$$
\begin{align}
& L(x, \text{expected}) \\
& = (\text{model}(x_0) - \text{expected})^2 \\
& = ( \; (x_1 \times k_2) \; - \text{expected})^2
\end{align}
$$

We're going to be using the chain rule, so let's pick our $f$ and $g$. I find it easier to think "inside to out", so, starting with $g$.

- $L(k_2) = f(g(k_2))$
- $g(k_2) = (x_1 \times k_2) - \text{expected}$
- $f(j) = j^2$

And from the chain rule above:

- $L'(k_2) = f'(g(k_2)) \; g'(k_2)$

(Remember that we're trying to find the partial derivative of $k_2$ in the model. To keep things easy to track, I'm naming $L'$'s argument $k_2$. Technically, we're defining a more general function, $L'(k)$, which is well-defined for all real numbers; but we'll only be evaluating it at $k_2$, so I find this notation easier to follow.)

So now let's put the pieces together. Let's start with $f$.

$$
\begin{array}{llr}
L'(k_2) & = f'(g(k_2)) \; g'(k_2) & \note{$f(x) = x^2$, so} \\
      && \note{$f'(x) = 2x$} \\[0.5em]
      & = 2(g(k_2)) \; g'(k_2)  & \note{expand $g(k_2)$} \\[0.5em]
      & = 2((x_1 \times k_2) - \text{expected}) \; g'(k_2)
\end{array}
$$

What's more, $g'(k_2)$ is easy: $g(k_2) = (x_1 \times k_2 - \text{expected})$, so its derivative with respect to $k_2$ is just $x_1$. So let's substitute that in our $L'$ equation:

$$
L'(k_2) = 2((x_1 \times k_2) - \text{expected}) \; x_1
$$

This is great: we've reduced the derivative of $k_1$ down to a function only involving $x_1$ and the expected value. We know the expected value from our training data, and all we need to do is to store $x_1$ during our forward pass phase and then plug it in during backpropagation.

Let's work through a concrete example. Taking the same LLM, let's pick some numbers:

- Our training data will be $\textit{input} = x_0 = 2$, $\textit{expected} = 36$
- Our initial guess for our trained parameters will be $k_1 = 3$, $k_2$ = 4.

We start with our forward pass:

$$
\begin{array}{llll}
\textbf{input}   \; &      & = \overbrace{x_0}^{2} & \\[0.5em]
\text{layer 1} \; & x_1  & = 2 \times \overbrace{k_1}^{3} = 6 & \\[0.5em]
\text{layer 2} \; & x_2  & = 6 \times \overbrace{k_2}^{4} = 24 & = \textbf{output}
\end{array}
$$

Now we'll compute the derivative of the loss function, $L'$. Because we had taken this derivative with respect to $k_2$, we'll calculate it at the value we currently have for $k_2$. When we encounter $x_1$, we'll just plug in the 6 from the forward pass.

$$
\begin{align}
L'(k_2) & = 2((x_1 \times k_2) - \overbrace{\text{expected}}^{36}) \; x_1 \\
        & = 2((\overbrace{x_1}^{6} \times k_2) - 36) \; \overbrace{x_1}^{6} \\
        & = 2(6 \times \overbrace{k_2}^{4}) - 36) \; 6 \\
        & = 2(6 \times 4 - 36) \; 6 \\
        & = -144
\end{align}
$$

We call this the {dfn}`loss gradient` for $k_2$. With it, we can now work our way up the math calls and figure out the loss gradient for $k_1$.

To do this, we'll take the partial derivative of the second layer, this time with respect to $k_1$. We'll by picking up where we did before in writing out $L$:

$$
\begin{align}
& L(x, \text{expected}) \\
& = ( \; (x_1 \times k_2) \; - \text{expected})^2 \\
& = ( \; ( (x_0 \times k_1) \times k_2) \; - \text{expected})^2
\end{align}
$$

Again we'll use the chain rule, this time setting:

- $g(k_1) = x_0 \times k_1$
- $f(j) = ( \, ( j \times k_2) \; - \text{expected} \, )^2$

Let's solve $L'(k_1)$:

$$
L'(k_1) & = f'(g(k_1)) \; g'(k_1)
$$

- $g(k_1)$ is trivial: $x_0 \times k_1$
- $g'(k_1)$ is also trivial: $x_0$
- $f'(j)$ is easy:

  $$
  \begin{align}
  f'(j) &= 2 \, f(j) \\
          &= 2 \, ( \, ( j \times k_2) \; - \text{expected} \, )
  \end{align}
  $$

If we work through our example above:

:::{warning} WRONG

This is all wrong. I need to work through it when I'm less tired.

<https://claude.ai/chat/a922720f-20da-4885-8eba-c936837d28c8>

WRONG:
$$
\begin{array}{llr}
L'(k_1) &= f'(g(k_1)) \, g'(k_1) & \note{expand $f'$} \\[0.5em]
        &= 2 \, ( \, ( g(k_1) \times k_2) \; - \text{expected} \, ) \, g'(k_1) & \note{expand $g$} \\[0.5em]
        &= 2 \, ( \, ( x_0 \times k_1 \times k_2) \; - \text{expected} \, ) \, g'(k_1) & \note{expand $g'$} \\[0.5em]
        &= 2 \, ( \, ( x_0 \times k_1 \times k_2) \; - \text{expected} \, ) \, x_0 \\[0.5em]
        &\note{fill in all the values:} \\[0.5em]
        &= 2 \, ( \, ( \overbrace{x_0}^{2} \times \overbrace{k_1}^{3} \times \overbrace{k_2}^{4}) \; - \overbrace{\text{expected}}^{36} \, ) \, \overbrace{x_0}^{2} \\[0.5em]
        &= 2 \, ( \, ( 2 \times 3 \times 4) \; - 36 \, ) \, 2 \\[0.5em]
\end{array}
$$

WRONG:
$$
\begin{array}{llr}
L'(k_1) &= f'(g(k_1)) \, g'(k_1) & \note{expand $g(k_1)$} \\[0.5em]
        &= f'(x_0 \times k_1) \, g'(k_1) & \note{expand $f'$} \\[0.5em]
        &= 2 \, ( \, ( k_1 \times k_2) \; - \text{expected} \, ) \, g'(k_1) & \note{expand $g'$}\\[0.5em]
        &= 2 \, ( \, ( k_1 \times k_2) \; - \text{expected} \, ) \, x_0 \\[0.5em]
        &\note{fill in all the values:} \\[0.5em]
        &= 2 \, ( \, ( \overbrace{k_1}^{3} \times \overbrace{k_2}^{4}) \; - \overbrace{\text{expected}}^{36} \, ) \, \overbrace{x_0}^{2} \\[0.5em]
        &= 2 \, ( 3 \times 4 - 36 ) \times 2
        &= -96
\end{array}
$$
:::
