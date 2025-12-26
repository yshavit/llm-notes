# Overview of LLMs

Before we get into the meat of things, I think it's useful to provide a 20,000-foot view of LLMs to help orient ourselves.

As you may have heard, LLMs are essentially "autocomplete on steroids": given a bunch of input text, they predict the next word. But how?

## How data flows through an LLM

:::{note} Technical terminology
This section will talk about vectors and tokens. If you don't come from a programming background, for now you can think of a vector as a list, and a token as a word in a sentence.
:::

LLMs encode pretty much all information as vectors. The overall flow is:

1. Turn the input text into a vector of vectors: the "outer" vector's elements each correspond to a token in the input, and each "inner" token encodes the LLM's knowledge about that token.
   {drawio}`images/intro/input-tokens`

2. Pass this vector-of-vector through a series of transformers. For now, you can just think of these as opaque boxes that contain the model's knowledge of language.

   Each transformer turns one vector-of-vectors into a different vector-of-vectors. The output vectors don't have any intuitive meaning.
   {drawio}`images/intro/intermediate-tokens`

3. Turn the last unintuitive vector-of-vectors into a vector that predicts the next token: each item in the vector corresponds to a token in the LLM's known vocabulary, and its value is the probability that it's the next token:
   {drawio}`images/intro/predictions`

If we're using the LLM in a chatbot, we'll then just take the most likely next-token, append it to the input text, and then loop again from the top.

{drawio}`images/intro/chatbot-loop`

## Training and inference

An LLM, like any machine learning model, has two basic modes:

(training-vs-inference)=

training
: The model is learning the values of its trained parameters. This is part of creating the model.

inference
: The model is using what it learned. This is the mode you interact with when you use the model.

## What's in a transformer?

I mentioned above that the model's transformers are opaque boxes that contain "the model's knowledge". The values that encode this knowledge are called {dfn}`learned parameters`; but where do these learned parameters come from, and what do they represent?

(vectors-are-nuance)=
LLMs use vectors to encode basically anything that has nuance. For example, the word "dog" has many meanings. It can be a noun or a verb; a pet, a service animal, or a hound of war; an ugly person or a scandalous person, either judgementally or affectionately ("you dog, you!"). It can mean some subtle thing that I don't even know how to think about, much less describe.

Vectors are how LLMs encode all of this information. The nuance of what "dog" means is encoded in a particular vector called the token embedding vector, which I'll talk about more in later chapters.

But what do these values actually represent? Basically: nothing that corresponds to human intuition. I've been saying that the values represent things like "dog can be a pet", but it's really more of "dog has a high value for property 6321 in the embedding vector", where property 6321 is... something which, in practice, tends to correlate with the right prediction for the next token. I find it helpful to think of it as "pet-ness" by way of analogy, but just remember that the analogy is imperfect.

So far, I've been talking about the word "dog" and its token embedding vector. But there are other pieces of information: the fact that "dog" is the ninth word in "the quick brown fox jumps over the lazy dog"; the fact that this is a common expression; the fact that referencing an animal in one part of the sentence makes it likely you'll reference another animal later; and so on. Each of these is a different vector, in a different part of the LLM. And again, each of these meanings is only an analogy.

The values for these properties are emergent, arising over many training rounds over a large body of text. Through the magic of partial derivatives and some other math tricks, the learned parameters naturally settle into useful values.

Gaining insight into what those values really "mean," and how we can understand or even audit them, is well outside the scope of this book. This is an area of active research (and is one of the things that Anthropic specifically works hard at).

There are a _lot_ of these learned parameters. A typical high-quality LLM (ChatGPT, Claude, etc) will have hundreds of billions of them. A small model that you can run on your laptop may have "only" 5 - 15 billion. Newer models are coming out with _trillions_.

(training-analogy)=

## An analogy for how training works

I'll describe training in more detail {ref}`later <introduction-to-training>`, but it may help demystify things a bit if I touch on it now. I mentioned above that the model's learned parameters are emergent properties. But how do they emerge, and how can they possibly mean anything if we didn't tell them what to mean?

An analogy may be helpful.

Imagine that a language's underlying structure can be represented by an intricate, stretchy fabric. Our goal is to find where that fabric's strong and weak strands are, and in particular to figure out the pattern of those strong and weak points. If we can do that, then when someone gives us a small piece of similar fabric, we can extrapolate a larger tapestry from it. Crucially, we don't know _how_ this fabric represents the language: we just know that it does. (More precisely: we just know that if we use it, we get useful results.)

To find those strands, we'll start by sprinkling a fine sand on the fabric. At this point, we still don't know anything about the fabric. But, randomness being what it is, some grains will have ended up slightly clumped near weak spots, causing those areas to very slightly, imperceptibly, sag.

Now we bump the surface softly on the left. This represents a round of training, and it moves the grains of sand in a not-quite-random way: They generally move rightward (this represents training bias), but more importantly, they'll also slide down a bit towards whatever sag in the fabric they're closest to.

Then we bump the surface softly on the right; and then on the top; and then the bottom, and every which way. Slowly but surely, the training biases cancel out, the effects of the sagging compound, and we get a sense of the fabric's composition. The weaker the spot in the fabric, the more the sand will accumulate and the more that area will sag.

In this metaphor, each sand-filled sag represents a learned parameter, and the amount of sand in it represents the parameter's value.

Note that the sags we find aren't the true structure of the fabric (or, by way of analogy, the language's structure). They're just a way to approximate it, which we discovered via a randomized process. If we were to clear the surface and start again with a fresh sprinkling or slightly different jostles, we might get a different set of sags. But both end results represent the same thing: an approximation of the fabric's true structure, which we can then use to generate more fabric.

## In summary

- everything is a vector or a matrix
- these vectors and matrices encode the nuances of human language
- each vector or matrix encodes a different aspect of those nuances
- the nuances aren't ones we'd understand; they're abstract mathematical properties that are only analogous to the kinds of categories we'd come up with
- it can't be stressed enough: _everything is a vector_ (or a matrix)

With that, let's get into it. We'll start with a refresher on what vectors and matrices are, and the handful of mathematical operations we'll need to use on them.
