# Download

Downloads GPT-2's model parameters from HuggingFace.

```bash
cargo run -p download -- [--size <124M | 355M | 774M | 1558M>]
```

This will download the files, and extract some metadata, to `data/<size>/*`.

I recommend you invoke it from the workspace root:

```bash
(
    cd "$(dirname "$(cargo locate-project --workspace --message-format plain)")"
    cargo run -p download
)
```