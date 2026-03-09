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

## Running

Very little care has been given to hardening the experience, especially with regard to working directories.

I recommend you run everything from this directory (`./simpllm` from the repo root).

Before you can run the LLM, you need to download training files from HuggingFace:

```bash
cargo run -p download -- # --size <124M | 355M | 774M | 1558M>
```

After that, from the same directory, you can run `simpllm`. This uses the home-grown, CPU-based tensor library in `simpllm-core`.

```bash
cargo run -p simpllm --release -- # --size <124M | 355M | 774M | 1558M>
```

- See `--help` for additional options (especially regarding logit sampling)
- `--release` isn't strictly required, but I highly suggest it. Otherwise, this thing _crawls_.

For a faster implementation that uses the same exact architecture but a real tensor library (not my toy implementation), you can run `fasterllm`:

```bash
cargo run -p fasterllm --release -- # --size <124M | 355M | 774M | 1558M>
```

The default configuration will still be CPU-based, but more optimized than the `simpllm-core` implementation. For even more speed, you should use CUDA on Windows, or Metal on macOS. See [the fasterllm readme](./crates/fasterllm/README.md) for details.

[my book]: http://llm-book.yuvalshavit.com/
