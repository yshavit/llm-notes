use gpt2weights::{read_nicely, FileNames, ModelPath};
use simpllm_core::bpe::Tokenizer;
use std::error::Error;

pub fn load_tokenizer(path: &ModelPath, file_names: &FileNames) -> Result<Tokenizer, Box<dyn Error>> {
    let vocab_bpe = read_nicely(path, &file_names.bpe_merges)?;
    let encoder_json = read_nicely(path, &file_names.bpe_encodings)?;
    Tokenizer::parse_vocab(vocab_bpe, encoder_json)
}
