use anyhow::{anyhow, Context, Result};
use clap::Parser;
use gpt2weights::{
    read_nicely, FileNames, Gpt2Size, HParams, ModelPath, ModelShape, Offsets, TensorFileOffsets, TransformerShape,
    SIMPLLM_METADATA_FILE,
};
use indicatif::{ProgressBar, ProgressStyle};
use memmap2::MmapOptions;
use safetensors::{Dtype, SafeTensors};
use std::fs;
use std::fs::File;
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

    let (offsets, transformer_shapes) = read_tensor_file(&model_path)?;

    let hf_config_stream = read_nicely(&model_path, HfModelFiles::Config.file_name())
        .map_err(|e| anyhow!("while reading {}: {e}", HfModelFiles::Config.file_name()))?;
    let h_params: HParams = serde_json::from_reader(hf_config_stream)?;

    let ms = ModelShape {
        transformer: transformer_shapes,
        h_params,
        tensor_offsets: offsets,
        file_names: FileNames {
            tensors: HfModelFiles::Tensors.file_name().to_string(),
            bpe_merges: HfModelFiles::BpeMerges.file_name().to_string(),
            bpe_encodings: HfModelFiles::BpeEncoding.file_name().to_string(),
        },
    };

    let ms_bytes = serde_json::to_vec_pretty(&ms)?;
    let out_path = model_path.path(SIMPLLM_METADATA_FILE);
    fs::write(&out_path, ms_bytes)?;
    println!("wrote {}", out_path.display());

    Ok(())
}

fn read_tensor_file(model_path: &ModelPath) -> Result<(TensorFileOffsets, Vec<TransformerShape>)> {
    let file = File::open(model_path.path(HfModelFiles::Tensors.file_name()))?;
    let header_bytes = unsafe { MmapOptions::new().map(&file)? };
    // Safetensors format is: 8 bytes for a header length, which is an LE u64; then the header; then the blocks.
    const SF_PRE_HEADER: usize = 8;

    let (sf_header_size, metadata) = SafeTensors::read_metadata(&header_bytes)?;
    let tensors_map = metadata.tensors();

    let mut model_file_index = TensorFileOffsets::default();
    let mut transformer_sizes: Vec<TransformerShape> = vec![];

    for (tensor_name, tensor_metadata) in tensors_map {
        let Some(parsed) = parse_tensor_name(&tensor_name, &mut model_file_index)? else {
            continue;
        };
        if tensor_metadata.dtype != Dtype::F32 {
            return Err(anyhow!("unexpected dtype in tensor {tensor_name}"));
        }
        *parsed.offsets = (
            tensor_metadata.data_offsets.0 + sf_header_size + SF_PRE_HEADER,
            tensor_metadata.data_offsets.1 + sf_header_size + SF_PRE_HEADER,
        );
        if let Some(n) = parsed.ffn_hidden_layer_bias {
            ensure_and_get(&mut transformer_sizes, n).ffn_hidden_layer_embed =
                tensor_metadata.shape[tensor_metadata.shape.len() - 1];
        }
    }

    Ok((model_file_index, transformer_sizes))
}

#[derive(Debug, Eq, PartialEq)]
struct TensorNameParse<'a> {
    offsets: &'a mut Offsets,
    ffn_hidden_layer_bias: Option<usize>,
}

fn ensure_and_get<I: Clone + Default>(list: &mut Vec<I>, num: usize) -> &mut I {
    if num >= list.len() {
        let missing_transformers_count = num - list.len() + 1; // num is 0-indexed
        let add = std::iter::repeat_n(I::default(), missing_transformers_count);
        list.extend(add);
    }
    &mut list[num]
}

fn parse_tensor_name<'a>(name: &str, builder: &'a mut TensorFileOffsets) -> Result<Option<TensorNameParse<'a>>> {
    //noinspection RsNeedlessLifetimes
    fn ok<'b>(offsets: &'b mut Offsets) -> Result<Option<TensorNameParse<'b>>> {
        // tiny helper to remove the Ok(Some(..)) noise
        Ok(Some(TensorNameParse {
            offsets,
            ffn_hidden_layer_bias: None,
        }))
    }

    match name {
        "ln_f.bias" => ok(&mut builder.final_norm.bias),
        "ln_f.weight" => ok(&mut builder.final_norm.scale),
        "wpe.weight" => ok(&mut builder.pos_embed),
        "wte.weight" => ok(&mut builder.tok_embed),
        name => {
            let mut splits = name.splitn(3, '.');

            let Some("h") = splits.next() else {
                anyhow::bail!("expected name to start with \"h.\"");
            };

            let num = splits.next().ok_or_else(|| anyhow!("expected \"h.<num\""))?;
            let num: usize = num.parse()?;

            let transformer = ensure_and_get(&mut builder.transformers, num);

            match splits.next() {
                Some("ln_1.bias") => ok(&mut transformer.before_attn_norm.bias),
                Some("ln_1.weight") => ok(&mut transformer.before_attn_norm.scale),

                Some("attn.bias") => Ok(None),
                Some("attn.c_attn.bias") => ok(&mut transformer.attn_qkv.bias),
                Some("attn.c_attn.weight") => ok(&mut transformer.attn_qkv.weight),
                Some("attn.c_proj.bias") => ok(&mut transformer.attn_wo.bias),
                Some("attn.c_proj.weight") => ok(&mut transformer.attn_wo.weight),

                Some("ln_2.bias") => ok(&mut transformer.before_ffn_norm.bias),
                Some("ln_2.weight") => ok(&mut transformer.before_ffn_norm.scale),

                Some("mlp.c_fc.bias") => ok(&mut transformer.ffn_hidden.bias),
                Some("mlp.c_fc.weight") => {
                    let offsets = &mut transformer.ffn_hidden.weight;
                    Ok(Some(TensorNameParse {
                        offsets,
                        ffn_hidden_layer_bias: Some(num),
                    }))
                }
                Some("mlp.c_proj.bias") => ok(&mut transformer.ffn_output.bias),
                Some("mlp.c_proj.weight") => ok(&mut transformer.ffn_output.weight),
                _ => Err(anyhow!("unexpected name")),
            }
        }
    }
}

async fn download_files(model_path: &ModelPath, check: bool) -> Result<()> {
    let repo = hf_repo(model_path.model());

    for model_file in HfModelFiles::iter().map(HfModelFiles::file_name) {
        let download_url = format!("{}/{}/resolve/main/{}", HF_BASE_URL, repo, model_file);
        let out_file = PathBuf::from(model_path.path(model_file));

        download_file(&download_url, &out_file, check)
            .await
            .with_context(|| format!("Failed to download {}", model_file))?;
    }
    Ok(())
}

async fn download_file(url: &str, output_path: &PathBuf, check: bool) -> Result<()> {
    let file_name = output_path.file_name().and_then(|n| n.to_str()).unwrap_or("unknown");

    // Check if the file already exists
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
