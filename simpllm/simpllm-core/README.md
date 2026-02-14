# simpllm-core

- [`bpe`](src/bpe): An implementation of a [byte-pair encoding][bpe] tokenizer.
- [`tensor`](src/tensor): Traits for tensor implementations.
- [`cputensor`](src/cputensor): The only implementation of the `tensor` traits — in this crate, anyway (see
  [`fasterllm`]) for another implementation).
- [`transformer`](src/transformer): The transformer block layers: norm, attention, and FFN.
- [`llm`](src/llm): Combines the input embedding and transformer layers into a single round of inference. Also includes
  a logit sampler for picking the next token based on the LLM's logits.

[bpe]: https://en.wikipedia.org/wiki/Byte-pair_encoding

[`fasterllm`]: ../crates/fasterllm