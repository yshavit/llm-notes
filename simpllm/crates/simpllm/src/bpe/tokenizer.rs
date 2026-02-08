use std::collections::HashMap;
use std::error::Error;
use std::fmt::Debug;
use std::io::{BufRead, BufReader, Read};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct Rank {
    value: u16,
}

pub struct Tokenizer {
    merge_rules: Vec<MergeRule>,

    tok_to_id: HashMap<String, u16>,
    id_to_tok: Vec<String>,

    bytes_to_unicode: [char; 256],
    unicode_to_bytes: HashMap<char, u8>,
}

#[derive(Debug)]
struct MergeRule(String, String);

impl Tokenizer {
    pub fn parse_vocab(merge_rules: impl Read, tok_to_id: impl Read) -> Result<Self, Box<dyn Error>> {
        let merge_rules = Self::parse_merge_rules(merge_rules)?;

        let tok_to_id: HashMap<String, u16> = serde_json::from_reader(tok_to_id)?;
        let id_to_tok = Self::build_id_to_tok(&tok_to_id)?;

        let bytes_to_unicode = Self::bytes_to_unicode();
        let unicode_to_bytes = bytes_to_unicode
            .into_iter()
            .enumerate()
            .map(|(b, c)| (c, b as u8))
            .collect();

        Ok(Self {
            merge_rules,

            tok_to_id,
            id_to_tok,

            bytes_to_unicode,
            unicode_to_bytes,
        })
    }

    pub fn encode(&self, text: &str) -> Vec<Rank> {
        // first, convert the text to bytes
        let as_bytes = text.as_bytes();
        // Now, convert each one of those bytes to its encoded form. We construct a fresh bytes_to_unicode, because
        let mut encoded: Vec<String> = as_bytes
            .into_iter()
            .map(|&b| self.bytes_to_unicode[b as usize])
            .map(String::from)
            .collect();

        // Do the merging
        let mut merge_rules = self.merge_rules.iter();
        while let Some(merge_rule) = merge_rules.next() {
            // look for matches
            let mut look_starting_at_idx = 0;
            loop {
                let encoded_segment = &encoded[look_starting_at_idx..];
                match Self::find_match(encoded_segment, merge_rule) {
                    None => break,
                    Some(index_within_segment) => {
                        let index_within_full = index_within_segment + look_starting_at_idx;

                        // remove the next word, and add it to this one
                        let _ = encoded.remove(index_within_full + 1);
                        encoded[index_within_full].push_str(&merge_rule.1);

                        // now we'll continue searching from here (inclusive)
                        look_starting_at_idx = index_within_full;

                        // Start from the top of the merge rules again, since the newly merged string may have been one
                        // of the rules we've already passed.
                        // Note that this won't affect the current loop; it'll just affect the next iteration of
                        // `while let Some(..) = merge_rules.next()`.
                        merge_rules = self.merge_rules.iter();
                    }
                }
            }
        }

        // Decode to ranks
        encoded
            .into_iter()
            .map(|e| Rank {
                value: self.tok_to_id[&e],
            })
            .collect()
    }

    pub fn decode_bytes(&self, ranks: &[Rank]) -> Vec<u8> {
        let ranks_encoded = ranks.iter().map(|r| {
            self.id_to_tok
                .get(r.rank())
                .ok_or_else(|| format!("rank lookup: {}", r.rank()))
        });
        ranks_encoded
            .into_iter()
            .flat_map(|enc| {
                let enc_bytes = match enc {
                    Ok(enc) => {
                        let maybe_bytes: Result<Vec<u8>, String> = enc
                            .chars()
                            .map(|c| {
                                self.unicode_to_bytes
                                    .get(&c)
                                    .map(|b| *b)
                                    .ok_or_else(|| format!("char lookup: {c}"))
                            })
                            .collect();
                        maybe_bytes
                    }
                    Err(err) => Err(err),
                };
                enc_bytes.unwrap_or_else(|err| format!("<??{err}??>").into_bytes())
            })
            .collect()
    }

    pub fn decode(&self, ranks: &[Rank]) -> String {
        let rank_bytes = self.decode_bytes(ranks);
        String::from_utf8_lossy(&rank_bytes).to_string()
    }

    pub fn is_eos(&self, rank: Rank) -> bool {
        self.id_to_tok
            .get(rank.rank())
            .map(|s| s == "<|endoftext|>")
            .unwrap_or(false)
    }

    fn find_match(text: &[String], rule: &MergeRule) -> Option<usize> {
        let mut text_iter = text.iter().enumerate().peekable();
        while let Some((idx, word)) = text_iter.next() {
            if word == &rule.0 {
                if let Some((_, next_word)) = text_iter.peek() {
                    if next_word == &&rule.1 {
                        return Some(idx);
                    }
                }
            }
        }
        None
    }

    fn build_id_to_tok(tok_to_id: &HashMap<String, u16>) -> Result<Vec<String>, Box<dyn Error>> {
        let mut with_optionals: Vec<Option<String>> = vec![None; tok_to_id.len()];
        for (tok, &id) in tok_to_id {
            let idx = id as usize;
            match with_optionals[idx] {
                None => with_optionals[idx] = Some(tok.to_string()),
                Some(_) => return Err(format!("duplicate entry at {idx}").into()),
            }
        }
        with_optionals
            .into_iter()
            .enumerate()
            .map(|(i, opt)| opt.ok_or_else(|| format!("no token at index {}", i).into()))
            .collect()
    }

    fn parse_merge_rules(vocab_file: impl Read) -> Result<Vec<MergeRule>, Box<dyn Error>> {
        let mut lines = BufReader::new(vocab_file).lines().enumerate();
        let Some((_, Ok(first_line))) = lines.next() else {
            return Err("couldn't read vocab.bpe".into());
        };
        if first_line != "#version: 0.2" {
            return Err("unexpected first line of vocab.bpe".into());
        }

        lines
            .map(|(line_no, line)| {
                let line = line?;
                let (a, b) = line
                    .split_once(" ")
                    .ok_or_else(|| format!("unexpected at {line_no} of vocab.bpe"))?;
                Ok(MergeRule(a.to_string(), b.to_string()))
            })
            .collect()
    }

    /// Adapted from <https://github.com/openai/gpt-2/blob/master/src/encoder.py>.
    fn bytes_to_unicode() -> [char; 256] {
        let mut result: [Option<char>; 256] = [None; 256];

        let mut bs: Vec<u8> = Vec::new();
        let mut cs: Vec<u32> = Vec::new();

        // Printable ASCII: ! to ~
        bs.extend(b'!'..=b'~');

        // Extended range: ¡ to ¬
        bs.extend(0xA1..=0xAC);

        // Extended range: ® to ÿ
        bs.extend(0xAE..=0xFF);

        // Copy bs to cs
        cs.extend(bs.iter().map(|&b| b as u32));

        let mut n = 0;
        for b in 0u8..=255 {
            if !bs.contains(&b) {
                bs.push(b);
                cs.push(256 + n);
                n += 1;
            }
        }

        // Fill in the result array
        for (b, c) in bs.into_iter().zip(cs.into_iter()) {
            result[b as usize] = Some(char::from_u32(c).unwrap());
        }

        // Convert to [char; 256], panicking if any are None
        result.map(|opt| opt.expect("bytes_to_unicode should map all 256 bytes"))
    }
}

impl Rank {
    pub fn rank(&self) -> usize {
        self.value as usize
    }
}

impl From<Rank> for u16 {
    fn from(value: Rank) -> Self {
        value.value
    }
}

impl From<usize> for Rank {
    fn from(value: usize) -> Self {
        Self {
            value: value.try_into().unwrap_or(0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::LazyLock;

    static TOKENIZER: LazyLock<Result<Tokenizer, String>> = LazyLock::new(|| {
        let vocab_bpe = include_str!("test_assets/vocab.bpe");
        let encoder_json = include_str!("test_assets/encoder.json");
        Tokenizer::parse_vocab(vocab_bpe.as_bytes(), encoder_json.as_bytes()).map_err(|e| format!("{e}"))
    });

    #[test]
    fn test_quick_brown_fox() {
        check("The quick brown fox jumps over the lazy dog.");
    }

    #[test]
    fn test_simple_word() {
        check("hello");
    }

    #[test]
    fn test_with_spaces() {
        check("hello world");
    }

    #[test]
    fn test_punctuation() {
        check("Hello, world!");
    }

    #[test]
    fn test_numbers() {
        check("The year is 2024.");
    }

    #[test]
    fn test_unicode() {
        check("café résumé");
    }

    #[test]
    fn test_newlines_and_tabs() {
        check("line1\nline2\ttab");
    }

    #[test]
    fn test_repeated_chars() {
        check("aaaaaa");
    }

    #[test]
    fn test_empty_string() {
        check("");
    }

    #[test]
    fn test_mixed_case() {
        check("ThIs Is MiXeD CaSe");
    }

    #[test]
    fn test_multiple_spaces() {
        check("hello    world");
    }

    #[test]
    fn test_leading_trailing_spaces() {
        check("  hello world  ");
    }

    #[test]
    fn test_only_punctuation() {
        check("!@#$%^&*()");
    }

    #[test]
    fn test_code_snippet() {
        check("fn main() { println!(\"Hello\"); }");
    }

    #[test]
    fn test_emoji() {
        check("Hello 👋 World 🌍");
    }

    #[test]
    fn test_control_characters() {
        check("\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f");
    }
    #[test]
    fn test_long_text() {
        check(
            "One morning, when Gregor Samsa woke from troubled dreams, he found himself transformed in his bed into a \
            horrible vermin. He lay on his armour-like back, and if he lifted his head a little he could see his brown \
            belly, slightly domed and divided by arches into stiff sections. The bedding was hardly able to cover it \
            and seemed ready to slide off any moment. His many legs, pitifully thin compared with the size of the rest \
            of him, waved about helplessly as he looked.",
        );
    }

    fn check(original_text: &str) {
        // round trip on my tokenizer
        let my_encoded_ranks: Vec<_> = {
            let my_tokenizer = TOKENIZER.as_ref().unwrap();
            let my_encoded = my_tokenizer.encode(original_text);

            let rank_to_string: String = my_tokenizer.decode(&my_encoded);
            assert_eq!(rank_to_string, original_text, "at round trip check");

            // return as usize
            my_encoded.iter().map(Rank::rank).collect()
        };

        // compare against tiktoken
        {
            let tiktoken_encoded: Vec<_> = tiktoken_rs::r50k_base_singleton()
                .encode_with_special_tokens(original_text)
                .into_iter()
                .map(|r| r as usize)
                .collect();
            assert_eq!(my_encoded_ranks, tiktoken_encoded, "at check against tiktoken");
        }
    }
}
