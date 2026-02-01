use simpllm::load::{ModelPath, load_model};
use std::io::stdin;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_name = "124M";

    let model_path = ModelPath::from(model_name);
    let model = load_model(&model_path)?;

    let tokenizer = tiktoken_rs::r50k_base_singleton();

    for line in stdin().lines() {
        let line = line.expect("error reading line. invalid utf-8?");
        let tok_indexes = tokenizer.encode_with_special_tokens(&line);

        print!("{tok_indexes:?} => ");

        let result_str = tokenizer.decode(tok_indexes);
        let result_str = result_str?;
        println!("{result_str}");
    }

    Ok(())
}
