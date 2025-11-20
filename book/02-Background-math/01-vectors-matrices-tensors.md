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

## Scalars, vectors, matrices, and tensors

For our purposes:

- A **scalar** is just a plain number, like 27.1.

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

:::{tip} Tensors more generally
:class: dropdown

In math and physics, tensors have other properties. We won't need them, and we'll also never use tensors above rank 3. So, you can basically think of a tensor as "like a matrix, but with three indices."
:::

## Math operations

There are two main math operations you'll need to know about. In both cases, the details aren't actually important: what's important is the shape of the inputs and outputs.

(matrix-math-summary)=
+++
dot products
: Combines two vectors into a single, scalar. Both vectors must be the same length.
  $$
  \mathbf{v} \cdot \mathbf{w} = \text{scalar number}
  $$

matrix multiplication:
: These combine two matrices into another matrix. The first matrix's column length has to be the second matrix's row length. The result has the same number of rows as the first matrix, and the same number of columns as the second.
  $$
        A_{\textcolor{lime}{\underline{a}} \times \textcolor{goldenrod}{b}}
  \cdot B_{\textcolor{goldenrod}{b} \times \textcolor{red}{\underline{c}}}
      = C_{\textcolor{lime}{\underline{a}} \times \textcolor{red}{\underline{c}}}
  $$
+++

All you really need to know is the shape of the math operations. But if you want to know how they actually work, that's the rest of this chapter.

:::{note} Details on matrix math
:class: dropdown

A {dfn}`dot product` combines two vectors of the same size (dimensionality) into a single number.

The two vectors are often represented as a horizontal vector $\cdot$ a vertical vector, but that's just a convention. It only matters that they have the same number of elements.

The dot product is simply the sum of terms, where each term the product of the two vectors' corresponding elements:

(dot-product-math)=
$$
\begin{bmatrix}
  \textcolor{red}{a} & \textcolor{forestgreen}{b} & \textcolor{goldenrod}{c}
\end{bmatrix}
\cdot
\begin{bmatrix}
  \textcolor{red}{\alpha} \\ \textcolor{forestgreen}{\beta} \\ \textcolor{goldenrod}{\gamma} \end{bmatrix}
= \textcolor{red}{a \cdot \alpha} + \textcolor{forestgreen}{b \cdot \beta} + \textcolor{goldenrod}{c \cdot \gamma}
$$

If the two vectors are normalized to have the same magnitude, the dot product specifies how aligned they are: higher values means more aligned. (This has a geometric interpretation, but it's not very important for LLMS. Just know that higher values mean more similar.)

---

In the {dfn}`matrix multiplication` of two matrices $A$ and $B$, each cell is the dot product of the corresponding row from $A$ and the corresponding column from $B$.

$$
\begin{aligned}
C &=
  \begin{bmatrix} \textcolor{blue}{A_{1,1}} & \textcolor{blue}{A_{1,2}} \\ \textcolor{lime}{A_{2,1}} & \textcolor{lime}{A_{2,2}} \end{bmatrix}
  \cdot
  \begin{bmatrix} \textcolor{red}{B_{1,1}} & \textcolor{goldenrod}{B_{1,2}} \\ \textcolor{red}{B_{2,1}} & \textcolor{goldenrod}{B_{2,2}} \end{bmatrix}
  \\[1.5em]
&=
  \begin{bmatrix}
    \begin{bmatrix} \textcolor{blue}{A_{1,1}} & \textcolor{blue}{A_{1,2}} \end{bmatrix}
    \cdot
    \begin{bmatrix} \textcolor{red}{B_{1,1}} \\ \textcolor{red}{B_{2,1}} \end{bmatrix}
    &
    \begin{bmatrix} \textcolor{blue}{A_{1,1}} & \textcolor{blue}{A_{1,2}} \end{bmatrix}
    \cdot
    \begin{bmatrix} \textcolor{goldenrod}{B_{1,2}} \\ \textcolor{goldenrod}{B_{2,2}} \end{bmatrix}
    \\[1em]
    \begin{bmatrix} \textcolor{lime}{A_{2,1}} & \textcolor{lime}{A_{2,2}} \end{bmatrix}
    \cdot
    \begin{bmatrix} \textcolor{red}{B_{1,1}} \\ \textcolor{red}{B_{2,1}} \end{bmatrix}
    &
    \begin{bmatrix} \textcolor{lime}{A_{2,1}} & \textcolor{lime}{A_{2,2}} \end{bmatrix}
    \cdot
    \begin{bmatrix} \textcolor{goldenrod}{B_{1,2}} \\ \textcolor{goldenrod}{B_{2,2}} \end{bmatrix}
  \end{bmatrix} \\[1em]
\end{aligned}
$$

Note:

(matrix-multiplication-notes)=

- The number of columns in $A$ must equal the number of rows in $B$. (This is just so that the dot products work).
- The resulting shape is $A_{\textcolor{blue}{a} \times \textcolor{pink}{b}} \cdot B_{\textcolor{pink}{b} \times \textcolor{red}{c}} = C_{\textcolor{blue}{a} \times \textcolor{red}{c}}$
- Matrix multiplication is not commutative: $AB \neq BA$ in general.

  For example, if we look at the first cell ($C_{1,1}$), it's:

  $$
  \begin{bmatrix} \textcolor{blue}{A_{1,1}} & \textcolor{salmon}{A_{1,2}} \end{bmatrix}
  \cdot
  \begin{bmatrix} \textcolor{steelblue}{B_{1,1}} \\ \textcolor{pink}{B_{2,1}} \end{bmatrix}
  $$

  If we commuted the matrices, this cell would be:

  $$
  \begin{bmatrix} \textcolor{steelblue}{B_{1,1}} & \textcolor{lime}{B_{1,2}} \end{bmatrix}
  \cdot
  \begin{bmatrix} \textcolor{blue}{A_{1,1}} \\ \textcolor{lawngreen}{A_{2,1}} \end{bmatrix}
  $$

  As you can see, only $\textcolor{blue}{A_{1,1}}$ and $\textcolor{steelblue}{B_{1,1}}$ appear in both.

For example:

$$
\begin{aligned}
 &\begin{bmatrix} \textcolor{blue}{1} & \textcolor{blue}{2} \\ \textcolor{lime}{3} & \textcolor{lime}{4} \end{bmatrix} \begin{bmatrix} \textcolor{red}{5} & \textcolor{goldenrod}{6} \\ \textcolor{red}{7} & \textcolor{goldenrod}{8} \end{bmatrix} \\[1.5em]
=&\begin{bmatrix}
\begin{bmatrix} \textcolor{blue}{1} & \textcolor{blue}{2} \end{bmatrix} \cdot \begin{bmatrix} \textcolor{red}{5} \\ \textcolor{red}{7} \end{bmatrix}
& \quad
\begin{bmatrix} \textcolor{blue}{1} & \textcolor{blue}{2} \end{bmatrix} \cdot \begin{bmatrix} \textcolor{goldenrod}{6} \\ \textcolor{goldenrod}{8} \end{bmatrix}
\\[1.25em]
\begin{bmatrix} \textcolor{lime}{3} & \textcolor{lime}{4} \end{bmatrix} \cdot \begin{bmatrix} \textcolor{red}{5} \\ \textcolor{red}{7} \end{bmatrix}
& \quad
\begin{bmatrix} \textcolor{lime}{3} & \textcolor{lime}{4} \end{bmatrix} \cdot \begin{bmatrix} \textcolor{goldenrod}{6} \\ \textcolor{goldenrod}{8} \end{bmatrix}
\end{bmatrix} \\[1.5em]
=&\begin{bmatrix} \textcolor{blue}{1} \cdot \textcolor{red}{5} + \textcolor{blue}{2} \cdot \textcolor{red}{7} & \textcolor{blue}{1} \cdot \textcolor{goldenrod}{6} + \textcolor{blue}{2} \cdot \textcolor{goldenrod}{8} \\ \textcolor{lime}{3} \cdot \textcolor{red}{5} + \textcolor{lime}{4} \cdot \textcolor{red}{7} & \textcolor{lime}{3} \cdot \textcolor{goldenrod}{6} + \textcolor{lime}{4} \cdot \textcolor{goldenrod}{8} \end{bmatrix} \\[1.5em]
=&\begin{bmatrix} 19 & 22 \\ 43 & 50 \end{bmatrix}
\end{aligned}
$$

:::
