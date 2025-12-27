# Refresher on Vectors and Matrices

Before we get into the meat of LLMs, let's do a quick refresher on vectors, matrices, and tensors.

This subject can take up whole chapters of a math text book, but we only need to know a few things:

- what scalars, vectors, and matrices are
- a tiny bit of what a tensor is
- how to combine vectors using dot products
- how to add matrices
- how to multiply matrices

If you know all that, feel free to skip this chapter.

## Scalars, vectors, matrices, and tensors

For our purposes:

- A **scalar** is just a plain number, like 27.1.

- A **vector** is just an ordered list things. A vector's elements are typically either numbers:

  $$
  \mathbf{v} = \begin{bmatrix}1 & 2 & 3 \end{bmatrix}
  $$

  ...or other vectors:

  $$
  \mathbf{v} =
  \begin{bmatrix}
  \begin{bmatrix}1 & 2 & 3 \end{bmatrix}
  & \begin{bmatrix}1 & 2 & 3 \end{bmatrix}
  \end{bmatrix}
  $$

- A **matrix** is a grid of numbers:

  $$
  M= \begin{bmatrix}4 & 5 & 6 \\ 7 & 8 & 9 \end{bmatrix}
  $$

- A **tensor** generalizes vectors and matrices.

We refer to items within a vector by its index, starting with 1: $\mathbf{v}_1 = 1$ in the above.

We refer to items within a matrix by its row and column, in that order: $M_{1,2} = 5$.

If we think of vectors as "an object with a single index" and matrices as "an object with two indexes", a tensor just abstracts that into N indices. That N is called the tensor's rank: a vector is a rank 1 tensor, and a matrix is a rank 2 tensor.

We call the number of elements in a vector its {dfn}`size`, or its {dfn}`dimensionality`. The terms are essentially interchangeable, though I find I tend to use "size" more when talking about the mechanics of math operations, and "dimensionality" more when talking about how much information the vector carries.

Lastly, we can treat a vector of size $d$ as a matrix of size $1 \times d$ or $d \times 1$.

$$
\underbrace{
  \begin{vmatrix}
  1 & 2 & 3
  \end{vmatrix}
}_{\text{size 3 vector}}
\longleftrightarrow
\underbrace{
  \begin{bmatrix}
  1 & 2 & 3
  \end{bmatrix}
}_{1 \times 3 \text{ matrix}}
\longleftrightarrow
\underbrace{
  \begin{bmatrix}
  1 \\ 2 \\ 3
  \end{bmatrix}
}_{3 \times 1 \text{ matrix}}
$$

:::{tip} Tensors more generally
:class: dropdown

In math and physics, tensors have other properties. We don't need them, so you can basically think of a tensor as "like a matrix, but with more than two indices."
:::

(matrix-math-details)=

## Math operations

### Overview

There are just a few operations we'll need to understand:

- Two that come up all the time (so make sure you understand them well!):
  - {dfn}`dot products` on vectors
  - {dfn}`matrix multiplication`
- A few simple ones that come up less frequently:
  - matrix {dfn}`transposition`
  - {dfn}`adding matrices` and {dfn}`multiplying a matrix by a scalar`

Note that all of these work on vectors or matrices, not higher-rank tensors. When we work with higher-rank tensors, we'll use some indices to slice those tensors into vectors or matrices, and then apply the above operations to those slices. For example, given a rank-3 tensor $X_{k,i,j}$, we can think of each $X_{k,\text{ }\dots}$ as a matrix, and then apply some matrix operation to each one.

(matrix-math-summary)=
+++
dot products
: Combines two vectors into a single scalar. Both vectors must be the same length.
  $$
  \mathbf{v} \cdot \mathbf{w} = \text{scalar number}
  $$

matrix multiplication:
: Combines two matrices into another matrix. The first matrix's column length has to be the second matrix's row length. The result has the same number of rows as the first matrix, and the same number of columns as the second.

  $$
  A_{ \underline{a} \times b }
  \cdot B_{ b \times \underline{c} }
  = C_{ \underline{a} \times \underline{c} }
  $$

  The expression $A \cdot B$ can also be written as just $AB$.
+++

:::{note} Pay attention to the shapes of these operations
I find that in most cases, I don't need to think about the details of these math operations (though we will need to when we look at the LLM's [algebraic reformulations]). What's more useful is the _shape_ of the operations. For example, if I have a vector of size $a$ and I need to turn it into a vector of size $b$, I know I'll need an $a \times b$ matrix:

$$
\begin{array}{cccl}
\mathbf{a} & \cdot & ? & = \mathbf{b} \\
A_{1 \times a} & \cdot & X_{? \times ?} & = B_{1 \times b} \\
A_{1 \times a} & \cdot & X_{a \times b} & = B_{1 \times b}
\end{array}
$$

Similarly, with dot products, the useful bit is usually just to remember that it turns two same-sized vectors into a single scalar.

[algebraic reformulations]: ../03-The-LLM/06-algebraic-reformulations.md
:::

+++
(matrix-transposition)=
transposition
: Swaps a matrix's rows and columns, which you can visualize as flipping along its ╲ diagonal. This is denoted as $A^T$.

  $$
  \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix}^T
  =
  \begin{bmatrix} 1 & 4 \\ 2 & 5 \\ 3 & 6 \end{bmatrix}
  $$
+++

adding two matrices
: We can add two matrices (or vectors) as long as they're the same size. This just means adding their corresponding elements:

  $$
  \begin{bmatrix}
  1 & 2 & 3 \\
  4 & 5 & 6
  \end{bmatrix}
  +
  \begin{bmatrix}
  10 & 20 & 30 \\
  40 & 50 & 60
  \end{bmatrix}
  = \begin{bmatrix}
  11 & 22 & 33 \\
  44 & 55 & 66
  \end{bmatrix}
  $$

multiplying by a scalar
: We can multiply a matrix (or vector) by a scalar, which just means applying the multiplication to each element:
  
  $$
  10 \cdot
  \begin{bmatrix}
  1 & 2 & 3 \\
  4 & 5 & 6
  \end{bmatrix}
  = \begin{bmatrix}
  10 & 20 & 30 \\
  40 & 50 & 60
  \end{bmatrix}
  $$

### Dot products

A {dfn}`dot product` combines two vectors of the same size (dimensionality) into a single number.

The two vectors are often represented as a horizontal vector on the left and a vertical vector on the right, but that's just a convention. It only matters that they have the same number of elements.

The dot product is simply the sum of terms, where each term the product of the two vectors' corresponding elements:

(dot-product-math)=
$$
\begin{bmatrix}
  \textcolor{red}{a} & \textcolor{limegreen}{b} & \textcolor{goldenrod}{c}
\end{bmatrix}
\cdot
\begin{bmatrix}
  \textcolor{red}{\alpha} \\ \textcolor{limegreen}{\beta} \\ \textcolor{goldenrod}{\gamma} \end{bmatrix}
= \textcolor{red}{a \cdot \alpha} + \textcolor{limegreen}{b \cdot \beta} + \textcolor{goldenrod}{c \cdot \gamma}
$$

If the two vectors are normalized to have the same magnitude, the dot product specifies how aligned they are: higher values means more aligned.

### Matrix multiplication

In the {dfn}`matrix multiplication` of two matrices $A$ and $B$, each cell $(i, j)$ is the dot product of the corresponding row $i$ from $A$ and column $j$ from $B$. This produces a thorough mixing of the two inputs: every row from $A$ gets combined with every column from $B$.

$$
\begin{aligned}
C &=
  \begin{bmatrix} \textcolor{steelblue}{A_{1,1}} & \textcolor{steelblue}{A_{1,2}} \\ \textcolor{limegreen}{A_{2,1}} & \textcolor{limegreen}{A_{2,2}} \end{bmatrix}
  \cdot
  \begin{bmatrix} \textcolor{red}{B_{1,1}} & \textcolor{goldenrod}{B_{1,2}} \\ \textcolor{red}{B_{2,1}} & \textcolor{goldenrod}{B_{2,2}} \end{bmatrix}
  \\[2.5em]
&=
  \begin{bmatrix}
    \begin{bmatrix} \textcolor{steelblue}{A_{1,1}} & \textcolor{steelblue}{A_{1,2}} \end{bmatrix}
    \cdot
    \begin{bmatrix} \textcolor{red}{B_{1,1}} \\ \textcolor{red}{B_{2,1}} \end{bmatrix}
    \quad
    &
    \begin{bmatrix} \textcolor{steelblue}{A_{1,1}} & \textcolor{steelblue}{A_{1,2}} \end{bmatrix}
    \cdot
    \begin{bmatrix} \textcolor{goldenrod}{B_{1,2}} \\ \textcolor{goldenrod}{B_{2,2}} \end{bmatrix}
    \\[2.25em]
    \begin{bmatrix} \textcolor{limegreen}{A_{2,1}} & \textcolor{limegreen}{A_{2,2}} \end{bmatrix}
    \cdot
    \begin{bmatrix} \textcolor{red}{B_{1,1}} \\ \textcolor{red}{B_{2,1}} \end{bmatrix}
    \quad
    &
    \begin{bmatrix} \textcolor{limegreen}{A_{2,1}} & \textcolor{limegreen}{A_{2,2}} \end{bmatrix}
    \cdot
    \begin{bmatrix} \textcolor{goldenrod}{B_{1,2}} \\ \textcolor{goldenrod}{B_{2,2}} \end{bmatrix}
  \end{bmatrix}
  \\[4em]
&=
  \begin{bmatrix}
    \textcolor{steelblue}{A_{1,1}} \cdot \textcolor{red}{B_{1,1}} + \textcolor{steelblue}{A_{1,2}} \cdot \textcolor{red}{B_{2,1}}
    &
    \quad
    \textcolor{steelblue}{A_{1,1}} \cdot \textcolor{goldenrod}{B_{1,2}} + \textcolor{steelblue}{A_{1,2}} \cdot \textcolor{goldenrod}{B_{2,2}}
    \\[1em]
    \textcolor{limegreen}{A_{2,1}} \cdot \textcolor{red}{B_{1,1}} + \textcolor{limegreen}{A_{2,2}} \cdot \textcolor{red}{B_{2,1}}
    &
    \quad
    \textcolor{limegreen}{A_{2,1}} \cdot \textcolor{goldenrod}{B_{1,2}} + \textcolor{limegreen}{A_{2,2}} \cdot \textcolor{goldenrod}{B_{2,2}}
  \end{bmatrix}
\end{aligned}
$$

When you multiply matrices:

(matrix-multiplication-notes)=

- The number of columns in $A$ must equal the number of rows in $B$. (This is just so that the dot products work).
- The resulting shape is $A_{ \underline{a} \times b } \cdot B_{ b \times \underline{c} } = C_{ \underline{a} \times \underline{c} }$
- Matrix multiplication is not commutative: $AB \neq BA$ in general.

  :::{hint} Why not?
  :class: simple dropdown no-margin

  If we look at the first cell ($C_{1,1}$), it's:

  $$
  \begin{bmatrix} A_{1,1} & A_{1,2} \end{bmatrix}
  \cdot
  \begin{bmatrix} B_{1,1} \\ B_{2,1} \end{bmatrix}
  $$

  If we commuted the matrices, this cell would be:

  $$
  \begin{bmatrix} B_{1,1} & B_{1,2} \end{bmatrix}
  \cdot
  \begin{bmatrix} A_{1,1} \\ A_{2,1} \end{bmatrix}
  $$

  As you can see, only $A_{1,1}$ and $B_{1,1}$ appear in both expressions. The same applies to each other cell, too.
  :::

For example:

$$
\begin{aligned}
 &\begin{bmatrix} \textcolor{steelblue}{1} & \textcolor{steelblue}{2} \\ \textcolor{limegreen}{3} & \textcolor{limegreen}{4} \end{bmatrix} \begin{bmatrix} \textcolor{red}{5} & \textcolor{goldenrod}{6} \\ \textcolor{red}{7} & \textcolor{goldenrod}{8} \end{bmatrix} \\[1.5em]
=&\begin{bmatrix}
\begin{bmatrix} \textcolor{steelblue}{1} & \textcolor{steelblue}{2} \end{bmatrix} \cdot \begin{bmatrix} \textcolor{red}{5} \\ \textcolor{red}{7} \end{bmatrix}
& \quad
\begin{bmatrix} \textcolor{steelblue}{1} & \textcolor{steelblue}{2} \end{bmatrix} \cdot \begin{bmatrix} \textcolor{goldenrod}{6} \\ \textcolor{goldenrod}{8} \end{bmatrix}
\\[1.25em]
\begin{bmatrix} \textcolor{limegreen}{3} & \textcolor{limegreen}{4} \end{bmatrix} \cdot \begin{bmatrix} \textcolor{red}{5} \\ \textcolor{red}{7} \end{bmatrix}
& \quad
\begin{bmatrix} \textcolor{limegreen}{3} & \textcolor{limegreen}{4} \end{bmatrix} \cdot \begin{bmatrix} \textcolor{goldenrod}{6} \\ \textcolor{goldenrod}{8} \end{bmatrix}
\end{bmatrix} \\[1.5em]
=&\begin{bmatrix} \textcolor{steelblue}{1} \cdot \textcolor{red}{5} + \textcolor{steelblue}{2} \cdot \textcolor{red}{7} & \textcolor{steelblue}{1} \cdot \textcolor{goldenrod}{6} + \textcolor{steelblue}{2} \cdot \textcolor{goldenrod}{8} \\ \textcolor{limegreen}{3} \cdot \textcolor{red}{5} + \textcolor{limegreen}{4} \cdot \textcolor{red}{7} & \textcolor{limegreen}{3} \cdot \textcolor{goldenrod}{6} + \textcolor{limegreen}{4} \cdot \textcolor{goldenrod}{8} \end{bmatrix} \\[1.5em]
=&\begin{bmatrix} 19 & 22 \\ 43 & 50 \end{bmatrix}
\end{aligned}
$$
