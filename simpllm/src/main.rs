use simpllm::load::{ModelPath, load_model};
use std::io::stdin;
use tiktoken_rs::Rank;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_name = "124M";

    let model_path = ModelPath::from(model_name);
    let model = load_model(&model_path)?.initialize();

    eprint!("Loading tokenizer...");
    let tokenizer = tiktoken_rs::r50k_base_singleton();
    eprintln!("Ready!");
    eprint!("> ");

    for line in stdin().lines() {
        let line = line.expect("error reading line. invalid utf-8?");
        let mut tok_indexes = tokenizer.encode_with_special_tokens(&line);

        // infer 10 times, hard-coded for now
        for _ in 0..10 {
            let result = model.apply(&tok_indexes)?;
            let last_row = result.num_rows() - 1;
            let result_rank = result.with_row([last_row, 0], |logits| {
                let mut all = Vec::from(logits);
                all.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                eprintln!("logits: {:?} ... {:?}", &all[..5], &all[all.len() - 5..]);

                let mut max_idx = 0;
                let mut max_val = logits[0];
                for i in 1..logits.len() {
                    if logits[i] > max_val {
                        max_val = logits[i];
                        max_idx = i;
                    }
                }
                max_idx
            });
            tok_indexes.push(result_rank as Rank);
            let result_str = tokenizer.decode(tok_indexes.clone())?;
            println!("inferred: {result_str}");
        }
    }

    Ok(())
}
