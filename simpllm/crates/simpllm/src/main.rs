use simpllm::bpe::Rank;
use simpllm::load::{ModelPath, load_model, load_tokenizer};
use simpllm::tensor::LogitSampler;
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
        let mut found_eos = false;
        for _ in 0..100 {
            let start_time = Instant::now();
            let result = model.apply(&tok_indexes)?;
            let result_rank: Rank = LogitSampler::new(result)
                .top_k(50)
                .top_prob(0.95)
                .temperature(0.9)
                .get()
                .into();

            durations.push(start_time.elapsed());

            if tokenizer.is_eos(result_rank) {
                found_eos = true;
                break;
            }

            tok_indexes.push(result_rank);

            stdout().write(&tokenizer.decode_bytes(&[result_rank]))?;
            let _ = stdout().flush();
        }

        if found_eos {
            println!(" <EOS>");
        } else {
            println!("...");
        }
        eprintln!("{durations:?}");
    }

    Ok(())
}
