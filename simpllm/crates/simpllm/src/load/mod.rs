mod metadata;
mod model;
mod path;
mod tokenizer;

pub use metadata::load_metadata;
pub use path::ModelPath;

pub type Result<T> = core::result::Result<T, Box<dyn std::error::Error>>;
pub use model::load_model;
pub use tokenizer::load_tokenizer;
