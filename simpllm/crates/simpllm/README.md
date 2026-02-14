# simpllm

An executable wrapper for [`simpllm-core`]. See `--help` for options.

This adds code to load model weights (downloaded by the [`download`] crate) and an input loop with a TUI for displaying
the results. I keep them it from `simple-core` so that the core can be free of external dependencies.

- I recommend running with `--release` (unless you're running in a debugger, of course): the speed increase is
  significant.
- The model weights must be in `./data/<size>/` relative to the current directory. Download them via the [`download`]
  executable.

The program will load the model weights, and then give you a `>` prompt.

## Windows

```powershell
Push-Location (Split-Path (cargo locate-project --workspace --message-format plain))
try {
    cargo run -p simpllm --release -- # [args ]
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
    cargo run -p simpllm --release -- # [args ]
)
```

[`download`]: ../download

[`simpllm-core`]: ../../simpllm-core
