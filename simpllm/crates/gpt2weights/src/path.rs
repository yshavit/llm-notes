use crate::Gpt2Size;
use std::env::current_dir;
use std::error::Error;
use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;

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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelFile {
    BpeMerges,
    BpeEncoder,
    MetadataJson,
    HParamsJson,
    TokEmbed,
    PosEmbed,
    Norm(NormFile, NormVariant),
    Transformer(TransformerN, TransformerComponent, WeightVariant),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NormVariant {
    Scale,
    Bias,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NormFile {
    Final,
    BeforeAttention(TransformerN),
    BeforeFfn(TransformerN),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TransformerN(pub usize);

impl TransformerN {
    fn prefix(self) -> String {
        format!("transformer.{:02}", self.0)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TransformerComponent {
    Qkv,
    AttnOutput,
    FfnHidden,
    FfnOutput,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WeightVariant {
    Weights,
    Bias,
}

impl ModelFile {
    pub fn to_string(self) -> String {
        fn s(s: &str) -> String {
            s.to_string()
        }
        match self {
            Self::BpeMerges => s("vocab.bpe"),
            Self::BpeEncoder => s("encoder.json"),
            Self::MetadataJson => s("metadata.json"),
            Self::HParamsJson => s("hparams.json"),
            Self::TokEmbed => s("tok_embed.bin"),
            Self::PosEmbed => s("pos_embed.bin"),
            ModelFile::Norm(which, variant) => {
                let which_str = match which {
                    NormFile::Final => s("final_norm"),
                    NormFile::BeforeAttention(n) => format!("{}.attn_norm", n.prefix()),
                    NormFile::BeforeFfn(n) => format!("{}.ffn_norm", n.prefix()),
                };
                let variant_str = match variant {
                    NormVariant::Scale => "scale",
                    NormVariant::Bias => "bias",
                };
                format!("{which_str}.{variant_str}.bin")
            }
            ModelFile::Transformer(n, component, variant) => {
                let prefix = n.prefix();
                let component_str = match component {
                    TransformerComponent::Qkv => "attn.qkv",
                    TransformerComponent::AttnOutput => "attn.output",
                    TransformerComponent::FfnHidden => "ffn.hidden",
                    TransformerComponent::FfnOutput => "ffn.output",
                };
                let variant_str = match variant {
                    WeightVariant::Weights => "weights",
                    WeightVariant::Bias => "bias",
                };
                format!("{prefix}.{component_str}.{variant_str}.bin")
            }
        }
    }
}

impl ModelPath {
    pub fn path(&self, file: ModelFile) -> PathBuf {
        let mut path_buf = self.unpack_dir();
        path_buf.push(file.to_string());
        path_buf
    }

    pub fn unpack_dir(&self) -> PathBuf {
        ["data", self.model.size(), "unpacked"].into_iter().collect()
    }
}

pub fn read_nicely(base: &ModelPath, file: ModelFile) -> Result<BufReader<File>, Box<dyn Error>> {
    let path = base.path(file);
    let raw = File::open(&path).map_err(|e| -> Box<dyn Error> {
        let pwd = current_dir()
            .map(|d| d.display().to_string())
            .unwrap_or_else(|_| "<?>".to_string());
        format!("failed to open {}: {e} (pwd={pwd})", path.as_path().display()).into()
    })?;
    Ok(BufReader::new(raw))
}
