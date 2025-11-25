# Putting it all together

{drawio}`Self-attention and the FFN combine to create a transformer|images/transformer/llm-flow-transformer`

:::{warning} WIP
TODO

In addition to whatever's in the book, make sure I cover:

1. Residual connections / skip connections: The attention output gets added back to the input (x + Attention(x)). This is crucial for training deep networks.
2. Layer normalization: Usually applied before or after attention (or both). This is important for training stability. Make sure I mention that epsilon is just to avoid dividing by zero.

> Why Not Use a Conditional instead of always adding epsilon?
> You're right that a conditional would work functionally, but there are a few reasons we don't do it that way:
>
> 1. Numerical stability (the main reason):
> Even when variance isn't exactly zero, it can be very close to zero (like 1e-10). Dividing by very small numbers causes numerical instability—you get huge values that can overflow or cause NaN (not-a-number) errors downstream.
> Adding epsilon ensures you're always dividing by at least epsilon (typically 1e-5 or 1e-6), which keeps everything numerically stable.
> 2. Differentiability:
> Conditionals create discontinuities in the function. For backpropagation, you need smooth gradients. The conditional variance == 0 ? epsilon : variance has a sharp jump, which is problematic for gradient-based optimization.
> Adding epsilon smoothly blends in, maintaining differentiability everywhere.
> 3. Hardware efficiency (your intuition):
> Yes, conditionals do cause branch divergence on GPUs/TPUs. Different threads taking different paths through a conditional can serialize execution and kill parallelism. Always adding epsilon avoids any branching.
:::
