use crate::Gpt2Size;
use std::error::Error;
use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ModelPath {
    model: Gpt2Size,
}

impl From<Gpt2Size> for ModelPath {
    fn from(model: Gpt2Size) -> Self {
        ModelPath { model }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelFile {
    TokenizerVocabBpe,
    TokenizerEncoderJson,
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
    fn to_string(self) -> String {
        fn s(s: &str) -> String {
            s.to_string()
        }
        match self {
            Self::TokenizerVocabBpe => s("vocab.bpe"),
            Self::TokenizerEncoderJson => s("encoder.json"),
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
        let mut segments = vec!["data", self.model.size(), "unpacked"];
        let file_name = file.to_string();
        segments.push(&file_name);
        segments.into_iter().collect()
    }
}

pub fn read_nicely(base: &ModelPath, file: ModelFile) -> Result<BufReader<File>, Box<dyn Error>> {
    let path = base.path(file);
    let raw = File::open(&path)
        .map_err(|e| -> Box<dyn Error> { format!("failed to open {}: {e}", path.as_path().display()).into() })?;
    Ok(BufReader::new(raw))
}
