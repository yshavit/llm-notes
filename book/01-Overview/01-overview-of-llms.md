# "An autocomplete on steroids"

:::{status} 2
:::

Before we get into the meat of things, I think it's useful to provide a 20,000-foot view of LLMs to help orient ourselves.

As you may have heard, LLMs are essentially "autocomplete on steroids": given a bunch of input text, they predict the next word. But how? The next two chapters will give a high-level explanation: this one will introduce the basic building blocks and terminology, and the next one will provide an outline of the LLM's overall architecture.

## How data flows through an LLM

:::{note} Technical terminology
This section will talk about vectors and tokens. If you don't come from a programming background, for now you can think of a vector as a list, and a token as a word in a sentence.
:::

The LLM starts with three main items:

- Its {dfn}`dictionary`, which is its set of known tokens (like {keyboard}`"a", "an", "aardvark", …`).
- Its {dfn}`parameters`, which are just numbers (_lots_ of them) that represent everything it's learned during training.
- Your prompt, which is just some text (we'll put images aside for this book).

The LLM takes those and does essentially three steps:

1. Turns the input text into a vector of vectors: the "outer" vector's elements each correspond to a token in the input, and each "inner" token encodes the LLM's surface-level knowledge about that token. This knowledge is encoded as numbers, but their values are opaque to us; each one just looks like random numbers.

   :::{drawio} images/intro/input-tokens
   :::

   For now, we can say that each of these inner vectors is  just a simple lookup based on the LLM's dictionary, like {keyboard}`a ⇒ [0.038, 0.583, 0.869, …]`. (That's not quite true, but I'll describe the details later.)

2. Next, the LLM uses its parameters to transform each inner vector to a different vector, again with an opaque meaning.

   :::{drawio} images/intro/intermediate-tokens
   :::

3. Finally, it turns each inner vector into a final vector, whose size is the size of the LLM's dictionary. Each slot in vector corresponds to one token in the LLM's dictionary, and the value represents how likely it is that that token is the next one in the sequence.

   Remember that this is still a vector of vectors. Each outer vector corresponds to the respective token in step 1; each inner vector represents the prediction of "for every token in my dictionary, how likely is it to be the next token."

   :::{drawio} images/intro/predictions
   :::

Of course, if our input tokens were {keyboard}`The` {keyboard}`quick` {keyboard}`brown` {keyboard}`fox`, then we don't really care about the next-token prediction for the first three tokens; we already have their actual next-tokens! ({keyboard}`The`'s is {keyboard}`quick`, and so on.) But for the last entry, we take the most likely next-token, and that's our autocomplete. We tack that onto the prompt, and then start the loop over.

:::{drawio} images/intro/chatbot-loop
:::

:::{tip} Why predict the tokens we already know?
:class: dropdown

The way the math works, we have to compute a prediction for each token; we can't only compute the last one. This actually does come in handy during training, when we're trying to hone the whole parameter set: it's useful to know what the LLM would have predicted comes after {keyboard}`The`, so that you can nudge it a bit closer to {keyboard}`quick`. At inference, a trick called KV caching --- you may know it as "prompt caching" --- lets us indeed only compute the last prediction.
:::

## Training and inference

An LLM, like any machine learning model, has two basic modes:

(training-vs-inference)=

training
: The model is learning the values of its trained parameters. This is part of creating the model.

inference
: The model is using what it learned. This is the mode you interact with when you use the model.

## What do all these numbers  mean?

I described a few of the values above as being opaque: both the calculated post-transform numbers, but also the LLM's parameters (which include the lookup table from each token to its vector of numbers). The parameters that make up the LLM's knowledge -- so, not the post-transform numbers, which combine that knowledge with the user's prompt -- are called {dfn}`learned parameters`; but where do these learned parameters come from, and what do they represent?

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

Imagine that a language's underlying structure can be represented by an intricate, stretchy fabric. Our goal is to find where that fabric's strong and weak strands are, and in particular to figure out the pattern of those strong and weak points. If we can do that, then when someone gives us a small piece of similar fabric, we can extrapolate a larger tapestry from it.

To find those strands, we'll start by sprinkling a fine sand on the fabric. At this point, we still don't know anything about the fabric. But, randomness being what it is, some grains will have ended up slightly clumped near weak spots, causing those areas to very slightly, imperceptibly, sag.

Now we bump the surface softly on the left. This represents a round of training, and it moves the grains of sand in a not-quite-random way: They generally move rightward (this represents training bias), but more importantly, they'll also slide down a bit towards whatever sag in the fabric they're closest to.

Then we bump the surface softly on the right; and then on the top; and then the bottom, and every which way. Slowly but surely, the training biases cancel out, the effects of the sagging compound, and we get a sense of the fabric's composition. The weaker the spot in the fabric, the more the sand will accumulate and the more that area will sag.

In this metaphor, each sand-filled sag represents a learned parameter, and the amount of sand in it represents the parameter's value.

Note that the sags we find aren't the true structure of the fabric (or, by way of analogy, the language's structure). They're just a way to approximate it, which we discovered via a randomized process. If we were to clear the surface and start again with a fresh sprinkling or slightly different jostles, we might get a different set of sags. But both end results represent the same thing: an approximation of the fabric's true structure, which we can then use to generate more fabric.

Next, I'll go into just a bit more detail on the LLM architecture, to make things just a bit more concrete, and give you a map of where this is all going. There won't be any math (yet!).
