# This book's structure

## This book is meant to be read front-to-back

The driving principle behind this book's organization is that you should be able to read it front-to-back. This means:

- The book assumes you don't know anything relating to machine learning (ML) or LLMs.
- If you do know something, you can always skip past it; but you should never have to jump to an appendix and then back to where you were.
- No footnotes

That said, I'll sometimes need to tease ahead to topics that I'll discuss in detail later. When I do, I'll try to give just enough context to make the current thing I'm explaining make sense. I'll provide cross-reference links where relevant, but you shouldn't need to click through to them.

Human learning being the way it is, you may still need to refer back to a section you've already read; "front-to-back" doesn't mean you shouldn't ever need to do that. But the book isn't organized around you having to jump around. In particular, I will abstain from appendices and footnotes.

## Organization

The book is organized into three parts:

1. **Introduction** (you are here), which includes a very high level overview of LLMs and a quick refresher on vectors and matrices
2. **The LLM**, which will walk you through the structure of an LLM from 0 to 60
3. **Training**, which will discuss how an LLM learns the values that power that structure
4. **Further reading**, which will talk about modern improvements to the LLM, as well as other, related ML technologies.

## Callouts

Throughout the book, I'll use callouts like this:

:::{note} Example
Some explanatory text.
:::

Some of these will be collapsed and are expandable; others are just visual blocks.

- If the callout is collapsed, it's optional; feel free to skip it. If you're like me, you won't — but just know it's not very important.
- If it's not collapsible, it's important, and you should read it.

## What I assume about you

This book assumes high school math. Maybe a bit more, but not much.

The most advanced math topic is vectors and matrices, and even for those, the book includes an overview of what you need to know. There is also a glancing blow of tensors, but again, I'll explain just what you need from those.

It's also helpful to have familiarity with derivatives, but you won't have to know the nitty-gritty.

That said, this book _will_ be getting into the specific math behind LLMs, so the more comfortable you are with math, the easier it'll likely be to follow along.

## The term "LLM"

LLMs — large language models — encompass a range of technologies. These can include models that generate text, but also translation tools, classification tools, and others. There are various architectures under the umbrella of LLMs, including BERT and others (I'll discuss some of these in @other-llm-models).

When most people talk about "LLMs" these days, they really mean the kind that can generate text and images, and specifically an LLM architecture called {dfn}`Generative Pre-Training`, or {dfn}`GPT`.

Following that colloquial usage, this book will use "LLM" and "GPT" interchangeably.
