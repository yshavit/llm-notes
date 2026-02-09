use crate::cputensor::LogitSampler;
use crate::run::Cli;
use crate::run::load::load_model;
use crate::run::tokenizer::load_tokenizer;
use crate::tensor::{Tensor, Tensor2D, TensorBackend};
use clap::Parser;
use gpt2weights::ModelPath;
use std::error::Error;
use std::io::{Write, stdin, stdout};
use std::time::Instant;

pub fn run_main<B: TensorBackend>() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();
    let model_path = ModelPath::from(cli.size);
    let model = load_model::<B>(&model_path)?.initialize();

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
        let mut remaining = cli.generate_limit;
        loop {
            match remaining.as_mut() {
                None => { /* nothing */ }
                Some(0) => {
                    println!("...");
                    break;
                }
                Some(remaining) => *remaining = remaining.saturating_sub(1),
            }

            let start_time = Instant::now();
            let result = model.apply(&tok_indexes)?;
            let result_rank = if cli.no_sampling {
                result.with_row([result.num_rows() - 1, 0], |row| {
                    // argmax
                    row.iter()
                        .enumerate()
                        .reduce(|acc, e| if e.1 > acc.1 { e } else { acc })
                        .map(|r| r.0)
                        .unwrap_or(0)
                })
            } else {
                LogitSampler::<B>::new(result)
                    .top_k(cli.sample_top_k.into())
                    .top_prob(cli.sample_top_p.into())
                    .temperature(cli.sample_temp.into())
                    .get()
                    .into()
            }
            .try_into()?;

            durations.push(start_time.elapsed());

            if tokenizer.is_eos(result_rank) {
                println!("<EOS>");
                break;
            }

            tok_indexes.push(result_rank);

            stdout().write(&tokenizer.decode_bytes(&[result_rank]))?;
            let _ = stdout().flush();
        }
        eprintln!("{durations:?}");
    }

    Ok(())
}
