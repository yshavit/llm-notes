use crate::Gpt2Size;
use std::error::Error;
use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelPath {
    model: Gpt2Size,
}

impl From<Gpt2Size> for ModelPath {
    fn from(model: Gpt2Size) -> Self {
        ModelPath { model }
    }
}

impl ModelPath {
    pub fn path(&self, file_name: impl Filename) -> PathBuf {
        let segments = ["data", self.model.size(), "unpacked", &file_name.get()];
        segments.iter().collect()
    }
}

pub trait Filename {
    fn get(&self) -> String;
}

impl Filename for &str {
    fn get(&self) -> String {
        self.to_string()
    }
}

impl<const R: usize> Filename for [&str; R] {
    fn get(&self) -> String {
        (*self).join("")
    }
}

pub fn read_nicely(path: &PathBuf) -> Result<BufReader<File>, Box<dyn Error>> {
    let raw = File::open(path)
        .map_err(|e| -> Box<dyn Error> { format!("failed to open {}: {e}", path.as_path().display()).into() })?;
    Ok(BufReader::new(raw))
}
