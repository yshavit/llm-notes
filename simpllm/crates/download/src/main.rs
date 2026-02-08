use anyhow::{anyhow, Context, Result};
use clap::Parser;
use gpt2weights::{
    Gpt2Size, ModelFile, ModelPath, ModelShape, NormFile, NormVariant, TransformerComponent, TransformerN,
    TransformerShape, WeightVariant,
};
use indicatif::{ProgressBar, ProgressStyle};
use safetensors::SafeTensors;
use std::fs;
use std::io::{stdout, Write};
use std::path::PathBuf;
use strum::IntoEnumIterator;
use strum_macros::EnumIter;
use tokio::io::AsyncWriteExt;

const HF_BASE_URL: &str = "https://huggingface.co";

#[derive(Copy, Clone, Debug, EnumIter)]
enum HfModelFiles {
    Config,
    BpeEncoding,
    BpeMerges,
    Tensors,
}

impl HfModelFiles {
    fn file_name(self) -> &'static str {
        match self {
            HfModelFiles::Config => "config.json",
            HfModelFiles::BpeEncoding => "vocab.json",
            HfModelFiles::BpeMerges => "merges.txt",
            HfModelFiles::Tensors => "model.safetensors",
        }
    }
}

/// Maps GPT-2 model size to HuggingFace repository name
fn hf_repo(size: Gpt2Size) -> &'static str {
    match size {
        Gpt2Size::Size124M => "openai-community/gpt2",
        Gpt2Size::Size355M => "openai-community/gpt2-medium",
        Gpt2Size::Size774M => "openai-community/gpt2-large",
        Gpt2Size::Size1558M => "openai-community/gpt2-xl",
    }
}

#[derive(Parser, Debug)]
#[command(name = "download")]
#[command(about = "Download GPT-2 model files")]
struct Args {
    /// Model size to download
    #[arg(long, short, default_value = "124M")]
    size: Gpt2Size,

    /// Check existing downloads and re-download if needed
    #[arg(long)]
    check_download: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    println!("Downloading GPT-2 {} model files from HuggingFace...", args.size.size());

    let model_path = ModelPath::from(args.size);
    download_files(&model_path, args.check_download).await?;

    println!(
        "\nDownload complete! HuggingFace files saved to {}/",
        download_dir(&model_path, "")
    );

    let model_shape = write_tensor_files(&model_path)?;

    for hf_file in HfModelFiles::iter() {
        let copy_to = match hf_file {
            HfModelFiles::Config => Some(ModelFile::HParamsJson),
            HfModelFiles::BpeEncoding => Some(ModelFile::BpeEncoder),
            HfModelFiles::BpeMerges => Some(ModelFile::BpeMerges),
            HfModelFiles::Tensors => None,
        };
        let Some(copy_to) = copy_to else {
            continue;
        };
        fs::copy(download_dir(&model_path, hf_file.file_name()), model_path.path(copy_to))?;
        println!("wrote {}", copy_to.to_string());
    }

    let model_shape_json = serde_json::to_vec_pretty(&model_shape)?;
    fs::write(model_path.path(ModelFile::MetadataJson), model_shape_json)?;

    Ok(())
}

fn download_dir(model_path: &ModelPath, file: &str) -> String {
    format!("data/{}/download/{file}", model_path.model().size())
}

fn write_tensor_files(model_path: &ModelPath) -> Result<ModelShape> {
    let tensor_bytes = fs::read(download_dir(model_path, HfModelFiles::Tensors.file_name()))?;
    let tensors_handle = SafeTensors::deserialize(&tensor_bytes)?;
    let mut tensor_names = tensors_handle.names();
    tensor_names.sort();

    fs::create_dir_all(model_path.unpack_dir())
        .with_context(|| format!("Failed to create directory {}", model_path.unpack_dir().display()))?;

    struct HiddenFfn {
        layer_num: usize,
        dim: usize,
    }
    let mut hidden_ffn_d: Vec<HiddenFfn> = vec![];

    let mut tensor_names = tensors_handle.names();
    tensor_names.sort();

    for name in tensor_names {
        match parse_tensor_name(name)? {
            None => {}
            Some(file) => {
                let tensor = tensors_handle.tensor(&name)?;
                print!("writing {} ({:?})... ", file.to_string(), tensor.shape());
                let _ = stdout().flush(); // ignore flush errors, not much we can do about 'em
                let data = tensor.data();

                fs::write(model_path.path(file), data)?;
                println!("ok");

                if let ModelFile::Transformer(n, TransformerComponent::FfnHidden, WeightVariant::Weights) = file {
                    let dim = *tensor.shape().last().unwrap();
                    hidden_ffn_d.push(HiddenFfn { layer_num: n.0, dim })
                }
            }
        }
    }

    hidden_ffn_d.sort_by(|a, b| a.layer_num.cmp(&b.layer_num));
    let transformer = hidden_ffn_d
        .into_iter()
        .map(|ffn| TransformerShape {
            ffn_hidden_layer_embed: ffn.dim,
        })
        .collect();

    Ok(ModelShape { transformer })
}

fn parse_tensor_name(name: &str) -> Result<Option<ModelFile>> {
    use gpt2weights::TransformerComponent::*;
    use ModelFile::*;

    match name {
        "ln_f.bias" => Ok(Some(Norm(NormFile::Final, NormVariant::Bias))),
        "ln_f.weight" => Ok(Some(Norm(NormFile::Final, NormVariant::Scale))),
        "wpe.weight" => Ok(Some(PosEmbed)),
        "wte.weight" => Ok(Some(TokEmbed)),
        name => {
            let mut splits = name.splitn(3, '.');

            let Some("h") = splits.next() else {
                anyhow::bail!("expected name to start with \"h.\"");
            };

            let num = splits.next().ok_or_else(|| anyhow!("expected \"h.<num\""))?;
            let num: usize = num.parse()?;
            let num = TransformerN(num);

            fn ok(f: ModelFile) -> Result<Option<ModelFile>> {
                Ok(Some(f))
            }

            match splits.next() {
                Some("ln_1.bias") => ok(Norm(NormFile::BeforeAttention(num), NormVariant::Bias)),
                Some("ln_1.weight") => ok(Norm(NormFile::BeforeAttention(num), NormVariant::Scale)),

                Some("attn.bias") => Ok(None),
                Some("attn.c_attn.bias") => ok(Transformer(num, Qkv, WeightVariant::Bias)),
                Some("attn.c_attn.weight") => ok(Transformer(num, Qkv, WeightVariant::Weights)),
                Some("attn.c_proj.bias") => ok(Transformer(num, AttnOutput, WeightVariant::Bias)),
                Some("attn.c_proj.weight") => ok(Transformer(num, AttnOutput, WeightVariant::Weights)),

                Some("ln_2.bias") => ok(Norm(NormFile::BeforeFfn(num), NormVariant::Bias)),
                Some("ln_2.weight") => ok(Norm(NormFile::BeforeFfn(num), NormVariant::Scale)),

                Some("mlp.c_fc.bias") => ok(Transformer(num, FfnHidden, WeightVariant::Bias)),
                Some("mlp.c_fc.weight") => ok(Transformer(num, FfnHidden, WeightVariant::Weights)),
                Some("mlp.c_proj.bias") => ok(Transformer(num, FfnOutput, WeightVariant::Bias)),
                Some("mlp.c_proj.weight") => ok(Transformer(num, FfnOutput, WeightVariant::Weights)),

                _ => Err(anyhow!("unexpected name")),
            }
        }
    }
}

async fn download_files(model_path: &ModelPath, check: bool) -> Result<()> {
    let repo = hf_repo(model_path.model());

    for model_file in HfModelFiles::iter().map(HfModelFiles::file_name) {
        let download_url = format!("{}/{}/resolve/main/{}", HF_BASE_URL, repo, model_file);
        let out_file = PathBuf::from(download_dir(model_path, model_file));

        download_file(&download_url, &out_file, check)
            .await
            .with_context(|| format!("Failed to download {}", model_file))?;
    }
    Ok(())
}

async fn download_file(url: &str, output_path: &PathBuf, check: bool) -> Result<()> {
    let file_name = output_path.file_name().and_then(|n| n.to_str()).unwrap_or("unknown");

    // Check if file already exists
    if output_path.exists() {
        if !check {
            println!("{}: file already exists", output_path.display());
            return Ok(());
        }
        // TODO: Implement MD5 checksum verification for check mode
        println!("{}: file already exists", output_path.display());
        return Ok(());
    }

    // Create parent directory if it doesn't exist
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("Failed to create directory {}", parent.display()))?;
    }

    // Download the file
    let client = reqwest::Client::new();
    let response = client.get(url).send().await.context("Failed to send request")?;

    if !response.status().is_success() {
        anyhow::bail!("HTTP error: {}", response.status());
    }

    let total_size = response.content_length().unwrap_or(0);

    // Create progress bar
    let pb = ProgressBar::new(total_size);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{msg}\n[{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")?
            .progress_chars("#>-"),
    );
    pb.set_message(format!("Downloading {}", file_name));

    // Create output file
    let mut file = tokio::fs::File::create(output_path)
        .await
        .context("Failed to create output file")?;

    // Download and write with progress
    let mut stream = response.bytes_stream();
    use futures_util::StreamExt;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("Failed to read chunk")?;
        file.write_all(&chunk).await.context("Failed to write to file")?;
        pb.inc(chunk.len() as u64);
    }

    pb.finish_with_message(format!("Downloaded {}", file_name));

    Ok(())
}
