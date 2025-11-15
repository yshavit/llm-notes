# Training

:::{warning} WIP
TODO

Let's dig very slightly into the details.

When a particular vector first starts out in the model, it's given random values. At this point, it doesn't matter what the values are; just that they're all different, and thus will react in different ways to the same input. Over various training rounds, the values converge to useful values.


At every training pass we'll have some input and a known output. For example, we can take the sentence "I like to ski when it snows" and break it into an input of "I like to ski when it" and an expected output of "snows".

We'll take this input and feed it into the model. The result will be all of the words in the model knows, each with the predicted probability that it's the next word. Ideally, for this input, the model should predict that "snows" has a high probability. Initially, the model is random, so it won't.

Instead, we'll have a probability for "snows" that's pretty far off from the expected value. We'll then go through the model and perform **backpropagation**, which basically asks: "for every parameter in the model, how would you have to adjust it, holding the rest of the parameters constant, for 'snows' to have a higher probability?" It then applies all of those learnings — and there you go, that's one jostle of the surface.

Over time, each of these jostles either reinforce each other or cancel each other out. That's just random at first, but as the valleys become deeper, they start to self-reinforce and become something the model has learned. Again, we don't know what the model's "valleys" really mean; we certainly don't know where they are. But the initial randomness plus repeated training rounds causes them to pop into existence through emergent behavior, and if we built our model well, the patterns those valleys describe will let us generate text.
:::
