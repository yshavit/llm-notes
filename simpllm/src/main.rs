use simpllm::load::{ModelPath, load_model};
use std::env;
use std::io::{Write, stdin, stdout};
use tiktoken_rs::Rank;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args().take(3).collect();
    let model_name = match args.as_slice() {
        [] | [_] => "124M", // only program name (or maybe nothing at all!)
        [_program_name, model_name] => &model_name,
        [..] => return Err("too many args".into()),
    };

    let model_path = ModelPath::from(model_name);
    let model = load_model(&model_path)?.initialize();

    eprint!("Loading tokenizer... ");
    let tokenizer = tiktoken_rs::r50k_base_singleton();
    eprintln!("found special tokens:");

    let mut end_of_text_tok = None;
    let mut special_toks: Vec<_> = tokenizer.special_tokens().into_iter().collect();
    special_toks.sort();
    for special_tok in special_toks {
        let encoded = tokenizer.encode_with_special_tokens(special_tok);
        eprintln!("- {special_tok} -> {encoded:?}");
        if special_tok == "<|endoftext|>" && encoded.len() == 1 {
            end_of_text_tok = Some(encoded[0]);
        }
    }

    eprintln!();
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
            let result_rank = result_rank as Rank;
            tok_indexes.push(result_rank);
            if end_of_text_tok.map(|eot| eot == result_rank).unwrap_or(false) {
                break;
            }

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
