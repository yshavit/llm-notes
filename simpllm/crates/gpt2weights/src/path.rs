use crate::Gpt2Size;
use std::env::current_dir;
use std::error::Error;
use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;

pub const SIMPLLM_METADATA_FILE: &str = "simpllm.json";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelPath {
    model: Gpt2Size,
}

impl ModelPath {
    pub fn model(&self) -> Gpt2Size {
        self.model
    }
}

impl From<Gpt2Size> for ModelPath {
    fn from(model: Gpt2Size) -> Self {
        ModelPath { model }
    }
}

impl ModelPath {
    pub fn path(&self, file: &str) -> PathBuf {
        let mut path_buf = self.unpack_dir();
        path_buf.push(file);
        path_buf
    }

    pub fn unpack_dir(&self) -> PathBuf {
        ["data", self.model.size()].into_iter().collect()
    }
}

pub fn read_nicely(base: &ModelPath, file: &str) -> Result<BufReader<File>, Box<dyn Error>> {
    let path = base.path(file);
    let raw = File::open(&path).map_err(|e| -> Box<dyn Error> {
        let pwd = current_dir()
            .map(|d| d.display().to_string())
            .unwrap_or_else(|_| "<?>".to_string());
        format!("failed to open {}: {e} (pwd={pwd})", path.as_path().display()).into()
    })?;
    Ok(BufReader::new(raw))
}
