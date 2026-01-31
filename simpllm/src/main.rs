use std::io::stdin;

fn main() {
    let tokenizer = tiktoken_rs::r50k_base_singleton();
    for line in stdin().lines() {
        let line = line.expect("error reading line. invalid utf-8?");
        let tok_indexes = tokenizer.encode_with_special_tokens(&line);

        print!("{tok_indexes:?} => ");

        let result_str = tokenizer.decode(tok_indexes);
        let result_str = result_str.unwrap();
        println!("{result_str}");
    }
}
