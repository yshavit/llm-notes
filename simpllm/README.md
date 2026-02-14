# simpllm

A fully-from-scratch implementation of GPT-2 LLM inference, which I wrote based off [my book].

Main crates:

- [`simpllm-core`](simpllm-core): The meat of the project: a full LLM, with virtually no external dependencies.
- `crates/`
    - [`simpllm`](crates/simpllm): An executable wrapper around `simpllm-core` that adds an input loop and a nice(-ish)
      TUI.
    - [`fasterllm`](crates/fasterllm): A faster executable, which replaces my hand-rolled tensor implementation with a
      real one.
    - [`download`](crates/download): A small utility for downloading GPT-2 model files from HuggingFace.
    - [`gpt2weights`](crates/gpt2weights): Shared types so that `download`, `simpllm`, and `fasterllm` agree on metadata
      and model locations.

[my book]: http://llm-book.yuvalshavit.com/
