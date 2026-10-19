# The LLM Architecture, Without Math

:::{status} 0
:::

By the end of this chapter, you'll have a good grasp on the general shape of a GPT 2.5-style LLM. You won't know the math, but you'll know the operations and how they mechanically fit together fairly concretely. If that's all you're looking for, this chapter may be all you need to sate your curiosity!

:::{tip} This chapter is optional
On the other hand, if you prefer to learn things from the bottom up, you may want to skip this chapter; the next section of the book, "The LLM Architecture", covers everything this chapter does but in greater detail.
:::

## Overview

TODO

## Some pseudo-math

Okay, I know I said we won't get into the math in this chapter; but I do at least need to tell you what the math operations are, and what shape they take. I'll keep it high level.

First, the cast of characters. We have three:

1. {dfn}`Scalars` are just numbers, like $1$ or $0.27$.
2. {dfn}`Vectors` are list of things --- often of scalars, but really of anything. For example, $\begin{bmatrix}1 & 2 & 3 \end{bmatrix}$.

   The vectors we'll see in our LLMs will all either be vectors of scalars, or vectors-of-vectors-of-scalars:

   $$
   \begin{bmatrix}\begin{bmatrix}1 & 2 & 3 \end{bmatrix} & \begin{bmatrix}4 & 5 & 6 \end{bmatrix} \end{bmatrix}
   $$
3. {dfn}`Matrices` are 2-dimensional grids of scalars:

   $$
   \begin{bmatrix}4 & 5 & 6 \\ 7 & 8 & 9 \end{bmatrix}
   $$

Next, we have a handful of operations:

1. {dfn}`Dot products` take two equal-length vectors of scalars, and turn them into a single scalar. We traditionally print the first matrix horizontally and the second vertically, but that doesn't actually matter.

   :::{drawio} images/no-math/dot-products
   :::

2. {dfn}`Matrix multiplication` takes two matrices, and produces a third. The first matrix’s number of columns has to equal the second matrix’s number of rows; and then the result has the same number of rows as the first matrix, and the same number of columns as the second. It sounds complicated, but visually it's not that bad:

   :::{drawio} images/no-math/matrix-mult
   :::

3. {dfn}`Transposition` takes a single matrix, and just swaps its rows and columns:

   :::{drawio} images/no-math/transpose
   :::

4. Finally, {dfn}`matrix addition` takes two equal-sized matrices and just adds their corresponding elements:

   :::{drawio} images/no-math/matrix-add
   :::

In this chapter, we won't worry ourselves with how dot products and matrix multiplication work; just their shapes are enough.

## Self-attention

TODO

(multi-head goes here)

## Feed-forward networks (FFNs)

TODO

## Stacked transformer blocks

TODO

## Residual connections

TODO

## Output logits

TODO

## Next up: *the math!*

Next up: the LLM architecture in all is gory details and math. I'll start with a refresher on what vectors and matrices are, and the handful of mathematical operations we'll need to use on them; then I'll get into what makes each one of these components tick, down to the multiplications and additions.
