# fasterllm

A version of [`simpllm`]. See that crate for usage notes. This replaces the unoptimized, hand-rolled tensors in
`simpllm` with [candle-rs], which is a real tensor library.

To get the most benefit, you should:

- build with `--release`
- use CUDA on Windows (if available) or Metal on macOS.

## Windows

- Ensure CUDA is installed.

    - To check, try getting the CUDA compiler's version:

        ```bash
        nvcc --version
        ```

    - If that doesn't, install CUDA per <https://developer.nvidia.com/cuda-downloads>

- When building or using `cargo run`, use `--features cuda`

```powershell
Push-Location (Split-Path (cargo locate-project --workspace --message-format plain))
try {
  cargo run -p fasterllm --release --features cuda
}
finally {
  Pop-Location
}
```

## macOs

No extra installation needed

- When building or using `cargo run`, use `--features metal`

```bash
(
    cd "$(dirname "$(cargo locate-project --workspace --message-format plain)")"
    cargo run -p fasterllm --release --features metal
)
```

[candle-rs]: https://github.com/huggingface/candle

[`simpllm`]: ../simpllm
