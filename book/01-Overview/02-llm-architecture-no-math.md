# The LLM Architecture, Without Math

:::{status} 0
:::

By the end of this chapter, you'll have a good grasp on the general shape of a GPT 2.5-style LLM. You won't know the math, but you'll know the operations and how they mechanically fit together fairly concretely. If that's all you're looking for, this chapter may be all you need to sate your curiosity!

:::{tip} This chapter is optional
On the other hand, if you prefer to learn things from the bottom up, you may want to skip this chapter; the next section of the book, "The LLM Architecture", covers everything this chapter does but in greater detail.
:::

## Goals for this chapter

This chapter is structured in four parts:

1. **The math, without the math**: I'll start with a mathless description of the basic operations. For this, we're not interested in the specifics of how the operations work, but just their shapes. It turns out that'll be enough to give a good sense of how data flows through.
2. **High-level overview**: Then, I'll give an overview of the components and where they slot into the architecture. This will be light on details; it's just to help give context for the next sections.
3. **The components**: Next, I'll dig into the details of the specific components. For each one, I'll use the mathless math to describe the component's parts and how data flows through them.
4. **How it fits together**: Finally, I'll revisit the high-level architecture, but now in more detail. This section will fill in some of the rationale and hand-waving from the high-level view. It'll also talk about the all-important {dfn}`prompt caching` that you may have heard about.

This chapter does assume you've read [the previous chapter](./01-overview-of-llms.md) for some of the fundamental terminology, but it doesn't assume anything more than that.

## The math, without the math

Even though this chapter will steer away from the details of the math, we do want to get a bit more concrete than hand-wavingly saying "and then some magical math happens." To help us strike a middle ground, it's useful to know a handful of mathematical concepts, and the _shape_ of how they fit together. We don't need to know particulars; you'll see what I mean by the shape momentarily.

### The cast of characters

We'll be working with three kinds of operands:

1. {dfn}`Scalars` are just numbers, like $1$ or $0.27$.
2. {dfn}`Vectors` are list of things --- often of scalars, but really of anything. For example, $\begin{bmatrix}1 & 2 & 3 \end{bmatrix}$.

   The vectors we'll see in our LLMs will all either be vectors of scalars, or vectors-of-vectors-of-scalars:

   $$
   \begin{bmatrix}\begin{bmatrix}1 & 2 & 3 \end{bmatrix} & \begin{bmatrix}4 & 5 & 6 \end{bmatrix} \end{bmatrix}
   $$

   We call the number of elements in a vector its {dfn}`dimensionality`.

3. {dfn}`Matrices` are 2-dimensional grids of scalars:

   $$
   \begin{bmatrix}4 & 5 & 6 \\ 7 & 8 & 9 \end{bmatrix}
   $$

   When we specify the size of a matrix, we write it as $m \times n$, where $m$ is the matrix's number of rows, and $n$ is the number of columns. The matrix above, for example, is $2 \times 3$.

We can also think of vectors as a special case of matrices, where one of the sides has length 1. (It doesn't matter which side we set to 1, but for our purposes, we'll always set the rows to 1 and the columns as $n$.)

(vector-or-matrix)=
:::{drawio} images/no-math/vector-or-matrix
:::

(how-we-use-scalars-and-vectors)=
:::{important} How we'll use these
Generally speaking, we'll use each of these in distinct ways:

- **Scalars** describe quantitative **how much** of something: think of it as being like asking "on a scale of 1 to 10, ..."
- **Vectors** describe the qualitative **how** or **what** of something: "how do you feel?" or "what are you thinking about?"
- **Matrices** are the brains of the LLM; they're how it thinks about the "how"/"what"-vectors that flow through it.
:::

### The operations

Next, we have a handful of operations on these operands:

(no-math-operations)=
1. {dfn}`Dot products` take two equal-length vectors of scalars, and turn them into a single scalar. We traditionally print the first matrix horizontally and the second vertically, but that doesn't actually matter.

   :::{drawio} images/no-math/dot-products
   :::

2. {dfn}`Matrix multiplication` takes two matrices, and produces a third. The first matrix’s number of columns has to equal the second matrix’s number of rows; and then the result has the same number of rows as the first matrix, and the same number of columns as the second. It sounds complicated, but visually it's not that bad:

   (no-math-matrix-multiplication)=
   :::{drawio} images/no-math/matrix-mult
   :::

3. {dfn}`Transposition` takes a single matrix, and just swaps its rows and columns:

   :::{drawio} images/no-math/transpose
   :::

4. Finally, {dfn}`matrix addition` takes two equal-sized matrices and just adds their corresponding elements:

   :::{drawio} images/no-math/matrix-add
   :::

In this chapter, we won't worry ourselves with how dot products and matrix multiplication actually work; just their shapes are enough.

## High-level overview

Let's build a progressively more detailed (but still very high-level) picture of the shape of our inference engine. This will help us get grounded in the overall direction, before we dig into the individual components.

### Space-station view

We'll start the ultra-high view: just the inputs, a black box of math, and outputs.

- The input is a vector of vectors.
  - Each element in the "outer" vector corresponds to one token in the input.
  - These are called the {dfn}`input embeddings`.
- The output is also a vector of vectors --- but with very different meanings for the inner vectors.
  - Each element in the outer vector is the prediction for the corresponding token's next-token.
  - These predictions are called {dfn}`logits`.

:::{drawio} images/no-math/high-level-1
:::

Let's look at the input embeddings and output logits in a bit more detail.

#### Input embeddings

Each input token gets translated into an input embedding, which is a $d$-dimensional vector. (The dimensionality $d$ just describes how much we know about each token. The LLM designers tune this: bigger numbers mean the LLM learns more nuance, but requires more compute power and memory.) It combines two pieces of information: the token, and its position within the input. Both of those pieces of information are just trivially looked up in a dictionary based on what the LLM learned during training.

For example, take the phrase:

> time after time

Let's say that our LLM has learned the following token values:

- $\text{time}  = \begin{bmatrix}-0.78 & 1.1 & 0.30 & -1.4 \dots\end{bmatrix}$
- $\text{after} = \begin{bmatrix}0.65 & -1.3 & 0.84 & 1.3 \dots\end{bmatrix}$
- ...

And the following position values:

- $1^{\text{st}}\text{ word}  = \begin{bmatrix}0.84 & -1.2 & 0.63 & 1.5 \dots\end{bmatrix}$
- $2^{\text{nd}}\text{ word}  = \begin{bmatrix}-0.59 & 1.6 & -1.4 & 0.71 \dots\end{bmatrix}$
- $3^{\text{rd}}\text{ word}  = \begin{bmatrix}1.2 & 0.43 & -0.87 & -0.16 \dots\end{bmatrix}$
- ...

Each token's input embedding just adds the vectors for token and value. So:

$$
\begin{array}{rcll}
\text{(first)}\texttt{ "time"}  & = & \begin{bmatrix}-0.78 & 1.1 & \dots\end{bmatrix}  & + \begin{bmatrix}0.84 & -1.2 & \dots\end{bmatrix} \\
                                & = & \begin{bmatrix}0.06 & -0.10 & \dots\end{bmatrix} & \\[1.5em]
\texttt{"after"}                & = & \begin{bmatrix}0.65 & -1.3 & \dots\end{bmatrix}  & + \begin{bmatrix}-0.59 & 1.6 & \dots\end{bmatrix} \\
                                & = & \begin{bmatrix}0.06 & -0.20 & \dots\end{bmatrix} & \\[1.5em]
\text{(second)}\texttt{ "time"} & = & \begin{bmatrix}-0.78 & 1.1 & \dots\end{bmatrix}  & + \begin{bmatrix}1.2 & 0.43 & \dots\end{bmatrix} \\
                                & = & \begin{bmatrix}0.42 & 1.53 & \dots\end{bmatrix}  &
\end{array}
$$

Note that the two instances of $\texttt{"time"}$ used the same token values, but each one used its corresponding (different) position values.

:::{tip} Terminology details
:class: simple dropdown
The token values (without position) are called {dfn}`token embeddings`, and the position values are {dfn}`position embeddings`. "Embedding" here just means "a vector that encodes what we know."

$$
\text{input embedding} = \text{token embedding} + \text{position embedding}
$$
:::

#### Output logits

The result of the LLM's inference is a vector of vectors: one vector per input token, whose value is a vector of logits that predicts that input's next token.

Each of those prediction vectors is a $v$-dimensional vector, where $v$ is the number of tokens the LLM knows about (very loosely, the number of words it knows). We imagine sorting the tokens alphabetically; then, the value of each value in this $v$-vector is how likely the corresponding token is to be the next token.

For example, let's say we had an LLM that knew only five words:

$$
\begin{bmatrix} \texttt{aardvark} & \texttt{cat} & \texttt{fox} & \texttt{stoat} & \texttt{wolf} \end{bmatrix}
$$

Then, if the $v$-vector of logits for a given token is:

$$
\begin{bmatrix} 1.6\%, & 2.2\%, & 93\%, & 1.4\%, & 1.8\% \end{bmatrix}
$$

...it means the LLM has calculated that there's a 93% chance that the next token is "fox", a 2.2% chance it's "cat", and so on.

Note that every token in the input gets a prediction. This is useful during training, but at inference we only care about the last prediction; we simply ignore the others.

#### In summary

:::{important} Important!
:class: simple
Did I lose you a bit with the math? Here's where to snap back.
:::

- Our inputs are a bunch of $d$-vectors, one per input. These encode what we trivially know about the input (its token and position).
  - $d$ is the {dfn}`embedding dimension` --- basically, how much information we can learn about any given token.
- Out outputs are a bunch of $v$-vectors, one per input. These encode our predictions for each input (its logits).
  - $v$ is the vocabulary size: how many tokens does this LLM know.

### Airplane-level view

Let's zoom in a bit and start to get some resolution into that "matrix math and stuff" stand-in. Inside that bubble are basically two main chunks: a series of {dfn}`transformer blocks`, followed by an output projection:

:::{drawio} images/no-math/high-level-2
:::

We'll cover the transformer blocks more in the next sections. For now, all you need to know is that each one turns a $d$-vector into a different, but similarly-shaped $d$-vector. The first transformer's $d$-vectors are just the input embeddings; after that, each transformer's output is the next one's input.

At the end of the chain of transformers, the contents of our original $d$-vector input embeddings have been changed quite a bit --- but they're still $d$-vectors. What we need, however, are $v$-vectors for the logits. How do we do that?

First, remember that we can [turn each vector into a matrix](#vector-or-matrix) where one of the sides is 1. So, let's turn our $d$-vectors into $1 \times d$, and our $v$-vectors into $1 \times v$. Now we just need a way to go from a $1 \times d$ matrix to a $v \times 1$ matrix:

:::{drawio} images/no-math/output-projection-unknown
:::

The [matrix multiplication described above](#no-math-matrix-multiplication) does exactly that. All we need is a $d \times v$ matrix, which we multiply our $d \times 1$ matrix by to get the $v \times 1$ matrix:

:::{drawio} images/no-math/output-projection
:::

The output projection is really just that simple!

### Bird's-eye view

:::{aside}
I know, some birds fly higher than some airplanes. Metaphors are hard!
:::

Before we go into each component in depth, let's look at the transformer blocks. Each consists of three elements:

- a {dfn}`self-attention` layer (often called just "attention")
- a {dfn}`feed-forward network` (FFN) layer
- {dfn}`residual connections`

:::{drawio} images/no-math/high-level-transformer
:::

As I mentioned above, these work to transform one vector of $d$-vectors into another vector of $d$-vectors: the transformer layer changes the input's contents, but not its shape.

We'll cover the FFN and self-attention in more detail below --- they're really the heart of the LLM, and where all the tricky computation happens.

Both the FFN and self-attention layer take $d$-vectors as input and produce $d$-vectors as output. The {dfn}`residuals` aren't transformation layers, but connections between them. They're very simple: after each transformation layer (FFN or attention) does its thing, the residual takes each resulting $d$-vector and adds its corresponding original input $d$-vector. That's it!

:::{drawio} images/no-math/residuals
:::

It turns out that these simple residuals are extremely important! Without them, the FFN and attention layers have to learn two separate things: the "core" knowledge (like "cats are animals") as well as how to apply that knowledge to the input. This tends to make them over-fit their data and cause "sharp corners" in the transformations, which in turn limits how many such layers you can stack. Since each transformer represents a unit of learning, limiting the number of transformers in the LLM means there's a ceiling on how smart the LLM can be.

Residuals take care of the "apply that knowledge to the input" aspect, allowing the FFN and attention layers focus on learning just their core knowledge. This eliminates the limit on stacking, which is what lets LLM vendors build the trillion-parameter LLMs that have become so powerful.

If you're interested in a deeper dive, [this YouTube video](https://www.youtube.com/watch?v=QgH9sr7G13Q) does a great job of walking through the visuals.

## The transformer components

### Feed-forward networks (FFNs)

TODO

### Self-attention

:::{caution}
TODO: this section describes the attention pairs backwards, I think!
:::

:::{tip} Tricky stuff ahead
There's a lot going on in this section, so it may take a couple read-throughs and tracing through the steps. I'll try to walk it through as carefully as I can, but just be warned: if this seems hard to understand, it's because it is!
:::

After tokenization, the first step on our input embedding's journey is through the {dfn}`self-attention` mechanism. This part of the LLM specializes in figuring out the relationships between words, in context. For example, it learns why I wrote "this part of the LLM" and not "this LLM the part of", or why "the child ate a pie" is a sentence about lunch, and not mutant, flesh-eating pastries.

To do this, we look at all the pairs of tokens in the sentence: `("the", "child")`, `("the", "ate")`, etc. Crucially, we want to consider each pair not in isolation, but in the general context of the prompt: where the words are in the prompt overall, how far away from each other they are, and so on. That contextual analysis is the heart of self-attention.

Because the LLM is trying to predict the next token, we ignore pairs where the first word in the pair comes later in the prompt than the second: so `("the", "child")` is in, but `("child", "the")` is out. Without this restriction, the model would learn not just to predict future tokens, but also to retroactively "rationalize" past tokens; this isn't useful, and just confuses the model.

:::{drawio} images/no-math/attention-generally
:::

Let's take a look at the inputs and outputs of this attention layer. The inputs are a vector of vectors, one $d$-dimensional vector for each of the input embeddings. The outputs will also be a vector of vectors, and to keep things simple, we'll have them all be $d$-dimensional as well.

:::{drawio} images/no-math/attention-flows
:::

So, what powers this transformation? There are lots of things that theoretically could do it, but it turns out that we can answer it efficiently by breaking the problem down to two smaller questions:

1. **How strong** is the relationship the first word and the second?
2. **What** can we learn about the relationship by looking at the second word?

:::{tip} What's in a name?
:class: simple
The first question is phrased, in LLM jargon, as how strongly the first word "attends to" the second --- hence "attention". The "self" in "self-attention" refers to the fact that the prompt is looking at its own, internal relationships.
:::

Note that "how strong" is a quantitative question, while "what" is a qualitative question. That means that the first question should be a scalar (recall [above](#how-we-use-scalars-and-vectors)) while the second is a vector.

:::{drawio} images/no-math/attention-step-1
:::

Let's focus on the "what" question first. How can we turn the input embedding vector into this "what"-vector? Well, if we think of each of these $d$-vectors as a $1 \times d$ or $d \times 1$ matrix, then we can use a $d \times d$ matrix to turn the embedding into the "what":

:::{drawio} images/no-math/attention-wv-1
:::

That's straightforward enough! Now let's turn to the "how strong" question. How can we compute a scalar that describes the relationship between the first word and the second? To do that, we'll break this question down further:

1. **How** does the first word act in relationships?
2. **How** does the second word act in relationships?

Both of these are qualitative question, so we can describe each as a vector. If we describe them as equal-length vectors, then we can use the [dot product](#no-math-operations) to get the "how strong" scalar.

:::{drawio} images/no-math/attention-step-2
:::

And we already know how to turn an embedding into a vector: we just need a $d \times d$ matrix:

:::{drawio} images/no-math/attention-step-3
:::

(We can trivially transpose a $d \times 1$ vector to a $1 \times d$, so the fact that the "2nd" vector is horizontal instead of vertical here doesn't matter.)

To recap:

- To answer "how strong is the relationship" (a scalar), we calculated the dot product of two vectors: one for "how does the first word act in relationships" and one for "how does the second word act in relationships". To get each of those, we multiplied the corresponding input embedding with a matrix.
- To answer "what can we learn about the relationship by looking at the second word" (a vector) we multiplied the second embedding by a matrix.

While there are lots of input embeddings (one per input token), we'll only work with these three matrices. So for example, every time we look at an input embedding and ask "how does this act when it's the first word in a relationship", we'll always use the same "green" matrix. This matrix learns everything about what it means to be the first word in a pair, across all possible first-words.

But now that we've calculated this scalar and this vector, what do we do with them?

:::{important} TODO
I need to finish this section
:::

(multi-head goes here)

## KV caching

TODO

## Summary

TODO

## Next up: *the real math!*

Next up: the LLM architecture in all is gory details and math. I'll start with a refresher on what vectors and matrices are, and the handful of mathematical operations we'll need to use on them; then I'll get into what makes each one of these components tick, down to the multiplications and additions.
