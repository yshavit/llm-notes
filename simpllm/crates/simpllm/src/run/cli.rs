use crate::run::OptionalNum;
use clap::Parser;
use gpt2weights::Gpt2Size;

/// Simple LLM implementation.
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct Cli {
    /// The model to use.
    #[arg(long, default_value = "124M")]
    pub(super) size: Gpt2Size,

    /// How many tokens to generate for a given prompt.
    ///
    /// Note that the total text (including initial prompt) will never be longer than the GPT-2 context window size,
    /// which is 1024.
    #[arg(long)]
    pub(super) generate_limit: Option<usize>,

    /// Top-k sampling parameter. Set to a number or "off" to disable.
    ///
    /// See --sample-temp for details.
    #[arg(long, default_value = "50", value_name = "NUM|off")]
    pub(super) sample_top_k: OptionalNum<usize>,

    /// Top-p (nucleus) sampling parameter. Set to a number or "off" to disable.
    ///
    /// See --sample-temp for details.
    #[arg(long, default_value = "0.95", value_name = "NUM|off")]
    pub(super) sample_top_p: OptionalNum<f32>,

    /// Temperature sampling parameter. Set to a number or "off" to disable.
    ///
    /// Along with --sample-top-k and --sample-top-p, this specifies how the model randomizes the tokens it generates.
    ///
    /// Temperatures > 1 flatten the randomization, such that tokens are picked from a more uniform distribution.
    /// Temperatures < 1 sharpen the randomization, such that the model is more confident about the probabilities.
    ///
    /// Once we have all the logits (each possible token, with its associated probability), we:
    ///
    /// • pick the top K tokens, discarding the rest
    /// • pick the most likely tokens until they add up to P, discarding the rest
    /// • apply temperature, to either flatten or sharpen the remaining probabilities
    /// • randomly pick the next token, weighed by its probability
    ///
    /// Any of these can be adjusted or disabled by the corresponding flags.
    #[arg(long, default_value = "0.9", value_name = "NUM|off")]
    pub(super) sample_temp: OptionalNum<f32>,

    /// Disable sampling and always pick the most likely token (greedy decoding).
    #[arg(long, conflicts_with_all = ["sample_top_k", "sample_top_p", "sample_temp"])]
    pub(super) no_sampling: bool,
}
