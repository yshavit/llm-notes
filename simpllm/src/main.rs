use simpllm::load::{ModelPath, load_model};
use std::io::{Write, stdin, stdout};
use tiktoken_rs::Rank;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_name = "124M";

    let model_path = ModelPath::from(model_name);
    let model = load_model(&model_path)?.initialize();

    eprint!("Loading tokenizer...");
    let tokenizer = tiktoken_rs::r50k_base_singleton();
    eprintln!("Ready!");

    let mut line = String::new();

    loop {
        eprint!("> ");
        line.clear();
        stdin().read_line(&mut line)?;
        let line = line.trim_end_matches(['\r', '\n']);
        if line == "exit" {
            break;
        }

        print!("{line}");
        let _ = stdout().flush();

        let mut tok_indexes = tokenizer.encode_with_special_tokens(&line);

        // infer 10 times, hard-coded for now
        for _ in 0..50 {
            let result = model.apply(&tok_indexes)?;
            let last_row = result.num_rows() - 1;
            let result_rank = result.with_row([last_row, 0], |logits| {
                let mut all = Vec::from(logits);
                all.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

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

            let result_str = tokenizer
                .decode(vec![result_rank as Rank])
                .unwrap_or_else(|_| "<??>".to_string());
            print!("{result_str}");
            let _ = stdout().flush();
        }
        println!("...");
    }

    Ok(())
}
