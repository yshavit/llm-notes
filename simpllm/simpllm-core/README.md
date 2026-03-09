# simpllm-core

The meat of the project! A fully from-scratch implementation of an LLM. The only two external dependencies (not counting
for tests) are [`rand`] for sampling logits, and [`rayon`] for parallel processing (for matrix math).

- [`bpe`](src/bpe): An implementation of a [byte-pair encoding][bpe] tokenizer.
- [`tensor`](src/tensor): Traits for tensor implementations.
- [`cputensor`](src/cputensor): A relatively simple, CPU-based implementation of the `tensor` traits. See
  [`fasterllm`] for another implementation based on an external, production-ready library.
- [`transformer`](src/transformer): The transformer block layers: norm, attention, and FFN.
- [`llm`](src/llm): Combines the input embedding and transformer layers into a single round of inference. Also includes
  a logit sampler for picking the next token based on the LLM's logits.

[bpe]: https://en.wikipedia.org/wiki/Byte-pair_encoding

[`fasterllm`]: ../crates/fasterllm

[`rand`]: https://crates.io/crates/rand

[`rayon`]: https://crates.io/crates/rayon
