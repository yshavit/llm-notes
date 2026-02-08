use crate::bpe::Tokenizer;
use crate::load::ModelPath;
use crate::load::path::read_nicely;

pub fn load_tokenizer(path: &ModelPath) -> super::Result<Tokenizer> {
    let vocab_bpe = read_nicely(&path.path("vocab.bpe"))?;
    let encoder_json = read_nicely(&path.path("encoder.json"))?;
    Tokenizer::parse_vocab(vocab_bpe, encoder_json)
}
