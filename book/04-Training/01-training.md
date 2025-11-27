(introduction-to-training)=

# Introduction to training

:::{warning} WIP
TODO

Let's dig very slightly into the details.

When a particular vector first starts out in the model, it's given random values. At this point, it doesn't matter what the values are; just that they're all different, and thus will react in different ways to the same input. Over various training rounds, the values converge to useful values.

At every training pass we'll have some input and a known output. For example, we can take the sentence "I like to ski when it snows" and break it into an input of "I like to ski when it" and an expected output of "snows".

We'll take this input and feed it into the model. The result will be all of the words in the model knows, each with the predicted probability that it's the next word. Ideally, for this input, the model should predict that "snows" has a high probability. Initially, the model is random, so it won't.

Instead, we'll have a probability for "snows" that's pretty far off from the expected value. We'll then go through the model and perform **backpropagation**, which basically asks: "for every parameter in the model, how would you have to adjust it, holding the rest of the parameters constant, for 'snows' to have a higher probability?" It then applies all of those learnings — and there you go, that's one jostle of the surface.

Over time, each of these jostles either reinforce each other or cancel each other out. That's just random at first, but as the valleys become deeper, they start to self-reinforce and become something the model has learned. Again, we don't know what the model's "valleys" really mean; we certainly don't know where they are. But the initial randomness plus repeated training rounds causes them to pop into existence through emergent behavior, and if we built our model well, the patterns those valleys describe will let us generate text.
:::

## Training considerations

:::{warning} TODO
move this to the training section
:::

### Causal masking

This improvement only applies during training.

Remember that our LLM will ultimately be used to auto-complete prompts. That means that at the point where it's making predictions, it won't have access to words after the input: they haven't been written yet!

{drawio}`At "Houston we have", we don't yet know "a problem"|images/attention/causal-attention`

In any machine learning model, it's important that the model trains the same way it'll be used during [inference](#training-vs-inference). This means that the attention weight for any word after the query token should always be 0. For example, given the training input "Houston, we have a problem", if our query token is "have", it shouldn't attend to "a" or "problem" at all; when it comes time for inference, it won't have access to them.

To do this, we just need to zero out all tokens after the query token. In our attention weight matrix, each row belongs to a corresponding query token (first row for the first token, etc.), so to zero out all tokens after each row's query token, we just need to zero out the upper-right triangle of the matrix. We'll then need to renormalize the remaining values, to reflect that they represent the full probability distribution that we care about:

$$
\begin{align}
\begin{bmatrix}
0.38 & 0.32 & 0.14 & 0.16 \\
0.09 & 0.37 & 0.24 & 0.30 \\
0.49 & 0.09 & 0.04 & 0.38 \\
0.51 & 0.25 & 0.06 & 0.18
\end{bmatrix}
& \rightarrow
\begin{bmatrix}
0.38 & \textcolor{gray}{0} & \textcolor{gray}{0} & \textcolor{gray}{0} \\
0.09 & 0.37 & \textcolor{gray}{0} & \textcolor{gray}{0} \\
0.49 & 0.09 & 0.04 & \textcolor{gray}{0} \\
0.51 & 0.25 & 0.06 & 0.18
\end{bmatrix} \\
& \rightarrow
\begin{bmatrix}
1.00 & \textcolor{gray}{0} & \textcolor{gray}{0} & \textcolor{gray}{0} \\
0.20 & 0.80 & \textcolor{gray}{0} & \textcolor{gray}{0} \\
0.79 & 0.15 & 0.06 & \textcolor{gray}{0} \\
0.51 & 0.25 & 0.06 & 0.18
\end{bmatrix}
\end{align}
$$

In practice, we can do this more easily by applying the mask a bit earlier. Rather than setting the appropriate attention weights to 0 and then renormalizing, we can set the attention _scores_ (before softmax) to $-\infty$. Softmax handles $-\infty$ by (a) transforming it to 0 and (b) ignoring it when calculating the other values. This is exactly the result we want, so applying this causal masking to the attention scores instead of weights lets us skip the post-mask renormalization.

### Dropout

Like causal masking, this improvement only applies during training.

The problem this improvement solves is one of over-fitting: learning parameters that are _too_ tightly bound to the data we train on, and thus don't generalize well. Since the ultimate goal of our LLM is to generate new, and ideally unique text, over-fitting is a real danger. We don't want "To be" to always complete as Hamlet's soliloquy.

Dropout solves this by reducing the model's dependency on specific attention patterns during training.

The approach is simple: randomly zero out some of the attention weights during each training step. This essentially deactivates those attentions, causing the model to lean more on others. Each training round picks a different set of weights to drop, so over time, all of the weights get trained without ever over-depending on any particular one.

The only gotcha is that we still want the weights to be properly scaled. In particular, we want each weight's expected value to be unchanged by dropout. To accomplish this, we multiply the surviving weights by a compensating factor based on the dropout rate.

:::{seealso} Expected value
:open: false
This is a statistical term that basically means, for a randomized value, what would its average be after an infinite number of iterations? For example, if you were to roll a 6-sided die forever, the average of those rolls would converge to $\frac{1 + 2 + 3 + 4 + 5 + 6}{6} = 3.5$.

If you randomly zero half of the rolls but double the others, these two effects cancel each other out. Even though individual rolls can now have values that the original couldn't (like 0 or 10), the expected value remains 3.5.
:::

For example, let's say we set dropout to 50% (this is a hyperparameter that's set during training; it's typically closer to the 10% - 20% range in the real world). This means:

- Each element has a 50% chance of being dropped
- Each surviving item will be doubled

$$
\begin{bmatrix}
0.38 & 0.32 & 0.14 & 0.16 \\
0.09 & 0.37 & 0.24 & 0.30 \\
0.49 & 0.09 & 0.04 & 0.38 \\
0.51 & 0.25 & 0.06 & 0.18
\end{bmatrix}
\rightarrow
\begin{bmatrix}
\textcolor{blue}{0.76} & \textcolor{gray}{0} & \textcolor{gray}{0} & \textcolor{gray}{0} \\
\textcolor{gray}{0} & \textcolor{blue}{0.74} & \textcolor{blue}{0.48} & \textcolor{blue}{0.60} \\
\textcolor{blue}{0.98} & \textcolor{gray}{0} & \textcolor{gray}{0} & \textcolor{blue}{0.76} \\
\textcolor{gray}{0} & \textcolor{gray}{0} & \textcolor{blue}{0.12} & \textcolor{gray}{0}
\end{bmatrix}
$$

Note that:

- After the dropping and compensating, each row no longer adds up to 1. The third row, for example, adds up to 1.74! This is fine: what's important is that each weight's expected value stays the same whether we do or don't use dropout.
- We're not dropping half of the elements in any particular row, or even in the matrix. Instead, each element independently gets dropped or not. In the example above, only one row had exactly half its weights dropped, and overall we dropped 9 elements instead of 8. In practice, there are enough training rounds, and the matrices are large enough, that the randomness averages out.
