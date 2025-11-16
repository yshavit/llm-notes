# Refresher on Vectors and Matrices

## Overview

Before we get into the meat of LLMs, let's do a quick refresher on vectors, matrices, and tensors.

This subject can take up whole chapters of a math text book, but we only need to know a few things:

- what a vector is
- what a matrix it
- a tiny bit of what a tensor is
- how to combine vectors using dot products
- how to multiply matrices

If you know all that, feel free to skip this chapter.

## What are vectors, matrices, and tensors?

For our purposes:

- A **vector** is just an ordered list of numbers:

  $$
  \mathbf{v} = \begin{bmatrix}1 & 2 & 3 \end{bmatrix}
  $$

- A **matrix** is a grid of numbers:

  $$
  M= \begin{bmatrix}4 & 5 & 6 \\ 7 & 8 & 9 \end{bmatrix}
  $$

- A **tensor** generalizes vectors and matrices into objects with N indices.

We refer to items within a vector by its index, starting with 1: $\mathbf{v}_1 = 1$ in the above.

We refer to items within a matrix by its row and column, in that order: $M_{1,2} = 5$.

If we think of vectors as "an object with a single index" and matrices as "an object with two indexes", a tensor just abstracts that into N indices. That N is called the tensor's rank: a vector is a rank 1 tensor, and a matrix is a rank 2 tensor.

For LLMs, we won't need anything higher than a rank 3 tensor.

:::{tip}
In math and physics, tensors have other properties. We won't need them, and we'll also never use tensors above rank 3. So, you can basically think of a tensor as "like a matrix, but with three indices."
:::

## Math operations

### Dot products

A dot product combines two vectors of the same size (dimensionality) into a single number.

The two vectors are often represented as a horizontal vector $\cdot$ a vertical vector, but that's just a convention. It only matters that they have the same number of elements.

The dot product is simply the sum of terms, where each term the product of the two vectors' corresponding elements:

(dot-product-math)=
$$
\begin{bmatrix}
  \color{red}{a} & \color{forestgreen}{b} & \color{goldenrod}{c}
\end{bmatrix}
\cdot
\begin{bmatrix}
  \color{red}{\alpha} \\ \color{forestgreen}{\beta} \\ \color{goldenrod}{\gamma} \end{bmatrix}
= \color{red}{a \cdot \alpha} + \color{forestgreen}{b \cdot \beta} + \color{goldenrod}{c \cdot \gamma}
$$

If the two vectors are normalized to have the same magnitude, the dot product specifies how aligned they are: higher values means more aligned. (This has a geometric interpretation, but it's not very important for LLMS. Just know that higher values mean more similar.)

### Matrix multiplication

To multiply matrices $A$ and $B$, each cell is the dot product of the corresponding row from $A$ and the corresponding column from $B$.

$$
\begin{aligned}
C &=
  \begin{bmatrix} \color{blue}{A_{1,1}} & \color{blue}{A_{1,2}} \\ \color{lime}{A_{2,1}} & \color{lime}{A_{2,2}} \end{bmatrix}
  \cdot
  \begin{bmatrix} \color{red}{B_{1,1}} & \color{goldenrod}{B_{1,2}} \\ \color{red}{B_{2,1}} & \color{goldenrod}{B_{2,2}} \end{bmatrix}
  \\[1.5em]
&=
  \begin{bmatrix}
    \begin{bmatrix} \color{blue}{A_{1,1}} & \color{blue}{A_{1,2}} \end{bmatrix}
    \cdot
    \begin{bmatrix} \color{red}{B_{1,1}} \\ \color{red}{B_{2,1}} \end{bmatrix}
    &
    \begin{bmatrix} \color{blue}{A_{1,1}} & \color{blue}{A_{1,2}} \end{bmatrix}
    \cdot
    \begin{bmatrix} \color{goldenrod}{B_{1,2}} \\ \color{goldenrod}{B_{2,2}} \end{bmatrix}
    \\[1em]
    \begin{bmatrix} \color{lime}{A_{2,1}} & \color{lime}{A_{2,2}} \end{bmatrix}
    \cdot
    \begin{bmatrix} \color{red}{B_{1,1}} \\ \color{red}{B_{2,1}} \end{bmatrix}
    &
    \begin{bmatrix} \color{lime}{A_{2,1}} & \color{lime}{A_{2,2}} \end{bmatrix}
    \cdot
    \begin{bmatrix} \color{goldenrod}{B_{1,2}} \\ \color{goldenrod}{B_{2,2}} \end{bmatrix}
  \end{bmatrix} \\[1em]
\end{aligned}
$$

Note:

(matrix-multiplication-notes)=

- The number of columns in $A$ must equal the number of rows in $B$. (This is just so that the dot products work).
- The resulting shape is $A_{\color{blue}{a} \times \color{pink}{b}} \cdot B_{\color{pink}{b} \times \color{red}{c}} = C_{\color{blue}{a} \times \color{red}{c}}$
- Matrix multiplication is not commutative: $AB \neq BA$ in general.

  For example, if we look at the first cell ($C_{1,1}$), it's:

  $$
  \begin{bmatrix} \color{blue}{A_{1,1}} & \color{salmon}{A_{1,2}} \end{bmatrix}
  \cdot
  \begin{bmatrix} \color{steelblue}{B_{1,1}} \\ \color{pink}{B_{2,1}} \end{bmatrix}
  $$

  If we commuted the matrices, this cell would be:

  $$
  \begin{bmatrix} \color{steelblue}{B_{1,1}} & \color{lime}{B_{1,2}} \end{bmatrix}
  \cdot
  \begin{bmatrix} \color{blue}{A_{1,1}} \\ \color{lawngreen}{A_{2,1}} \end{bmatrix}
  $$

  As you can see, only $\color{blue}{A_{1,1}}$ and $\color{steelblue}{B_{1,1}}$ appear in both.

#### Example

$$
\begin{aligned}
 &\begin{bmatrix} \color{blue}{1} & \color{blue}{2} \\ \color{lime}{3} & \color{lime}{4} \end{bmatrix} \begin{bmatrix} \color{red}{5} & \color{goldenrod}{6} \\ \color{red}{7} & \color{goldenrod}{8} \end{bmatrix} \\[1.5em]
=&\begin{bmatrix}
\begin{bmatrix} \color{blue}{1} & \color{blue}{2} \end{bmatrix} \cdot \begin{bmatrix} \color{red}{5} \\ \color{red}{7} \end{bmatrix}
& \quad
\begin{bmatrix} \color{blue}{1} & \color{blue}{2} \end{bmatrix} \cdot \begin{bmatrix} \color{goldenrod}{6} \\ \color{goldenrod}{8} \end{bmatrix}
\\[1.25em]
\begin{bmatrix} \color{lime}{3} & \color{lime}{4} \end{bmatrix} \cdot \begin{bmatrix} \color{red}{5} \\ \color{red}{7} \end{bmatrix}
& \quad
\begin{bmatrix} \color{lime}{3} & \color{lime}{4} \end{bmatrix} \cdot \begin{bmatrix} \color{goldenrod}{6} \\ \color{goldenrod}{8} \end{bmatrix}
\end{bmatrix} \\[1.5em]
=&\begin{bmatrix} \color{blue}{1} \cdot \color{red}{5} + \color{blue}{2} \cdot \color{red}{7} & \color{blue}{1} \cdot \color{goldenrod}{6} + \color{blue}{2} \cdot \color{goldenrod}{8} \\ \color{lime}{3} \cdot \color{red}{5} + \color{lime}{4} \cdot \color{red}{7} & \color{lime}{3} \cdot \color{goldenrod}{6} + \color{lime}{4} \cdot \color{goldenrod}{8} \end{bmatrix} \\[1.5em]
=&\begin{bmatrix} 19 & 22 \\ 43 & 50 \end{bmatrix}
\end{aligned}
$$
