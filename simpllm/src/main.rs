use simpllm::bpe::Rank;
use simpllm::load::{ModelPath, load_model, load_tokenizer};
use std::env;
use std::io::{Write, stdin, stdout};
use std::time::Instant;

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
    let tokenizer = load_tokenizer(&model_path)?;
    eprintln!("found special tokens:");

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

        let mut tok_indexes = tokenizer.encode(&line);

        // infer 10 times, hard-coded for now
        let mut durations = Vec::new();
        for _ in 0..10 {
            let start_time = Instant::now();
            let result = model.apply(&tok_indexes)?;
            durations.push(start_time.elapsed());

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
            let result_rank: Rank = result_rank.into();
            if tokenizer.is_eos(result_rank) {
                break;
            }

            tok_indexes.push(result_rank);

            stdout().write(&tokenizer.decode_bytes(&[result_rank]))?;
            let _ = stdout().flush();
        }

        println!("...");
        eprintln!("{durations:?}");
    }

    Ok(())
}
