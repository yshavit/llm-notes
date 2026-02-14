# Introduction

:::{status} 2
:::

## What is this book, and who's it for?

"You don't really understand something until you can explain it."

This book is my attempt to synthesize my understanding of how LLMs work. It's based on my reading of [_Build a Large Language Model (From Scratch)_ by Sebastian Raschka][Raschka], as well as a lot of back and forth with AI chatbots to help me through the things I didn't understand.

I wrote this book for myself, because there's no better way to make sure you've learned something than to try to explain it. But it's my hope that others may find it useful as well.

## Feedback encouraged!

The bottom of the main nav (either the left pane, or the {keyboard}`≡` icon at the top bar, depending on your screen) includes a feedback form. This is an anonymous Google Sheets form. I don't track your email when you submit, and you don't need to be logged in.

I welcome any corrections, comments or questions. Please make sure to include the page and chapter you were on, as the form won't include them.

## The term "LLM"

LLMs --- large language models --- encompass a range of technologies. These include models that generate text, but also translation tools, classification tools, and others.

There are various architectures under the LLM umbrella, such as BERT (I'll cover some of these in @other-llm-models). But when most people talk about "LLMs", they really mean the ones that can generate text and images --- and specifically, an LLM architecture called {dfn}`Generative Pre-trained Transformer`, or {dfn}`GPT`.

Following that colloquial usage, this book will use "LLM" and "GPT" interchangeably.

## Organization

### Parts in the journey

I find it useful to think about LLMs in three hierarchical perspectives:

(pedagogical-hierarchy)=

1. The fundamental concepts
2. Algebraic reformulations of those concepts
3. The actual implementation

This book will primarily focus on the first two perspectives. It leaves the third essentially untouched, though I wrote [an implementation of a GPT-2 LLM][implementation] based on this book. (Let me know if you'd like me to tie this implementation more closely to the book!)

For more implementation details, you should refer to resources like [Sebastian Raschka's _Build a Large Language Model (From Scratch)_][Raschka] or [Hugging Face's course] (which I haven't read, but I hear good things about).

:::{note} This is not standard terminology
The way I break down these perspectives --- and in particular, the separation between fundamental concepts and algebraic reformulations --- isn't standard. Most texts combine the concepts and algebraic formulations, which makes for a more streamlined description, but one that I find harder to follow.

If you read other materials on LLMs, just be aware that they'll likely combine perspectives 1 and 2 into just a single "here's what's going on".
:::

The book is organized into four parts:

1. **Introduction** (you are here), which includes a very high level overview of LLMs and a quick refresher on vectors and matrices
2. **The LLM**, which will walk you through the architecture of an LLM from 0 to 60
3. **Training**, which will discuss how an LLM learns the values that drive that architecture
4. **Further reading**, which will talk about modern improvements to the LLM, as well as other, related ML technologies.

[implementation]: https://github.com/yshavit/llm-notes/tree/main/simpllm
[Raschka]: https://www.manning.com/books/build-a-large-language-model-from-scratch
[Hugging Face's course]: https://huggingface.co/learn/llm-course/chapter1/1

### This book is meant to be read front-to-back

The driving principle behind this book's organization is that you should be able to read it front-to-back. This means:

- The book assumes you don't know anything about machine learning (ML) or LLMs.
- If you do know something, you can always skip past it; but you should never have to jump ahead then back to where you were.
- In particular, this means no footnotes or appendices.

That said, I'll sometimes need to tease ahead to topics that I'll discuss in detail later. When I do, I'll try to give just enough context to make the current thing I'm explaining make sense. I'll provide cross-reference links where relevant, but you don't need to click through to them.

(Of course, human learning being the way it is, you may still need to refer back to a section you've already read to remind yourself of it. Basically: yes to having to flipping back, no to having to flipping ahead.)

### Callouts

Throughout the book, I'll use callouts like this:

:::{note} Example
Some explanatory text.
:::

Some of these will be collapsed and are expandable; others are just visual blocks.

- If the callout is collapsed, it's optional; feel free to skip it. If you're like me, you won't --- but just know it's not very important.
- If it's not collapsible, it's important, and you should read it.

### What I assume about you

This book assumes high school math. Maybe a bit more, but not much.

The most advanced math topic is vectors and matrices, and even for those, the book includes an overview of what you need to know. There is also a glancing blow of tensors, but again, I'll explain just what you need from those.

It's also helpful to have familiarity with derivatives, but you won't have to know the nitty-gritty.

That said, this book _will_ be getting into the specific math behind LLMs, so the more comfortable you are with math, the easier it'll likely be to follow along.

## Contributions

The source for this book is on [my GitHub][gh]. Please feel free to suggest corrections there, especially if I got something factually wrong.

[gh]: https://github.com/yshavit/llm-notes
