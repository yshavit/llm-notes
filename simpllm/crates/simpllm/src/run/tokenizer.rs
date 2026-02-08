use crate::bpe::Tokenizer;
use gpt2weights::path::{ModelPath, read_nicely};
use std::error::Error;

pub fn load_tokenizer(path: &ModelPath) -> Result<Tokenizer, Box<dyn Error>> {
    let vocab_bpe = read_nicely(&path.path("vocab.bpe"))?;
    let encoder_json = read_nicely(&path.path("encoder.json"))?;
    Tokenizer::parse_vocab(vocab_bpe, encoder_json)
}
