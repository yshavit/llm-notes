use crate::run::cli::Cli;
use crate::run::load::load_model;
use crate::run::simple_tui::Ui;
use crate::run::tokenizer::load_tokenizer;
use clap::Parser;
use crossterm::event;
use crossterm::event::{Event, KeyCode, KeyModifiers};
use gpt2weights::ModelPath;
use simpllm_core::bpe::Rank;
use simpllm_core::cputensor::LogitSampler;
use simpllm_core::tensor::TensorBackend;
use std::error::Error;
use std::io::{Write, stdin, stdout};
use std::time::{Duration, Instant};

pub fn run_main<B: TensorBackend>() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();
    let model_path = ModelPath::from(cli.size);
    let (model_loader, model_shape) = load_model::<B>(&model_path)?;
    let model = model_loader.initialize();

    eprint!("Loading tokenizer... ");
    let tokenizer = load_tokenizer(&model_path, &model_shape.file_names)?;
    eprintln!("<eos> is {}", model.eos());

    eprintln!();
    eprintln!("Ready!");
    eprintln!();

    let logit_sampler = if cli.no_sampling {
        None
    } else {
        Some(
            LogitSampler::default()
                .top_k(cli.sample_top_k.into())
                .top_prob(cli.sample_top_p.into())
                .temperature(cli.sample_temp.into()),
        )
    };

    let mut line = String::new();

    loop {
        line.clear();
        eprint!("\x1b[3;95m>\x1b[0m \x1b[3;36m");
        stdin().read_line(&mut line)?;
        eprint!("\x1b[0m");
        let line = line.trim_end_matches(['\r', '\n']);
        if line.is_empty() || line == "exit" {
            break;
        }
        let original_line_len = line.len();

        // Move back to the beginning of the line the user just hit enter from.
        print!("\x1b[1A\r");
        let _ = stdout().flush();

        let mut tok_indexes = tokenizer.encode(line);
        let mut durations = Vec::new();
        let mut remaining = cli.generate_limit;
        let mut generated = line.as_bytes().to_vec();
        let mut ctx = model.new_inference_context();
        let mut ui = Ui::new()?;
        let mut infer_on: &[Rank] = &tok_indexes;
        let mut single_rank: [Rank; 1];
        loop {
            if event::poll(Duration::from_millis(0))?
                && let Event::Key(key) = event::read()?
                && key.code == KeyCode::Char('c')
                && key.modifiers.contains(KeyModifiers::CONTROL)
            {
                generated.extend("\x1b[3;35m<user canceled inference>\x1b[0m".as_bytes());
                break;
            }

            match remaining.as_mut() {
                None => { /* nothing */ }
                Some(0) => {
                    generated.extend("\x1b[3;35m<max tokens reached>\x1b[0m".as_bytes());
                    break;
                }
                Some(remaining) => *remaining = remaining.saturating_sub(1),
            }

            let start_time = Instant::now();
            let result_rank = model.apply(infer_on, &mut ctx, &logit_sampler)?;
            single_rank = [result_rank];
            infer_on = &single_rank;

            durations.push(start_time.elapsed());

            if result_rank == model.eos().into() {
                break;
            }

            tok_indexes.push(result_rank);
            generated.extend(&tokenizer.decode_bytes(&[result_rank]));
            let generated_str = String::from_utf8_lossy(&generated);
            ui.inference(&generated_str, &durations, tok_indexes.len())?;
        }
        drop(ui);

        let generated_str = String::from_utf8_lossy(&generated);
        let (prompt, inference) = generated_str.split_at(original_line_len);
        println!("\x1b[3;36m{prompt}\x1b[0m{inference}");
        println!();
    }

    Ok(())
}
