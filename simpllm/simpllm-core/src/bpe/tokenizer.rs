use crate::bpe::gpt_tok_format::gpt2_bpe_char_to_byte;
use std::collections::HashMap;
use std::error::Error;
use std::fmt::Debug;
use std::io::{BufRead, BufReader, Read};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct Rank {
    value: u16,
}

type Tok = Vec<u8>;

pub struct Tokenizer {
    merge_rules: Vec<MergeRule>,

    tok_to_id: HashMap<Tok, u16>,
    id_to_tok: Vec<Tok>,
}

#[derive(Debug)]
struct MergeRule(Tok, Tok);

impl Tokenizer {
    pub fn parse_vocab(merge_rules: impl Read, tok_encoding_lines: impl Read) -> Result<Self, Box<dyn Error>> {
        let gpt2_to_byte = gpt2_bpe_char_to_byte();
        let merge_rules = Self::parse_merge_rules(merge_rules, &gpt2_to_byte)?;

        let id_to_tok: Vec<_> = BufReader::new(tok_encoding_lines)
            .lines()
            .map(|gpt2_line| {
                // convert the line from GPT-2's format to a Tok
                gpt2_line.map(|line| line.chars().map(|ch| gpt2_to_byte[&ch]).collect::<Vec<_>>())
            })
            .collect::<Result<_, _>>()?;
        let tok_to_id: HashMap<Tok, u16> = HashMap::from_iter(
            id_to_tok
                .iter()
                .enumerate()
                .map(|(idx, tok)| (tok.to_owned(), idx as u16)),
        );

        Ok(Self {
            merge_rules,

            tok_to_id,
            id_to_tok,
        })
    }

    pub fn encode(&self, text: &str) -> Vec<Rank> {
        // First, convert the text to Toks. Each u8 in the text just gets wrapped into a single-element Vec<u8>, which
        // is the Tok.
        let mut encoded: Vec<Tok> = text.as_bytes().iter().map(|&b| vec![b]).collect();

        // Merge the Toks.
        // Look for matches (MYSTMD::BPE::MERGE START)
        let mut merge_rules = self.merge_rules.iter();
        while let Some(merge_rule) = merge_rules.next() {
            let mut look_starting_at_idx = 0;
            loop {
                let segment = &encoded[look_starting_at_idx..];
                let Some(idx_within_segment) = Self::find_match(segment, merge_rule) else {
                    break;
                };
                let index_within_full = idx_within_segment + look_starting_at_idx;

                // Merge the words by removing the next word and adding it to this one.
                let _ = encoded.remove(index_within_full + 1);
                encoded[index_within_full].extend(merge_rule.1.clone());

                // Search for more matches of the merge_rule within the input, starting at the
                // index we just merged.
                look_starting_at_idx = index_within_full;

                // Reset the merge rules, since the newly merged string may have been one of
                // the rules we've already passed.
                // Note that this won't affect the current loop; it'll just affect the next
                // iteration of while let Some(..) = merge_rules.next()`.
                merge_rules = self.merge_rules.iter();
            }
        }
        // MYSTMD::BPE::MERGE END

        // Decode to ranks
        encoded
            .into_iter()
            .map(|e| Rank {
                value: self.tok_to_id[&e],
            })
            .collect()
    }

    pub fn decode_bytes(&self, ranks: &[Rank]) -> Vec<u8> {
        ranks
            .iter()
            .map(|r| self.id_to_tok.get(r.rank()).expect("rank lookup error"))
            .flat_map(|tok| tok.iter())
            .copied()
            .collect()
    }

    pub fn decode(&self, ranks: &[Rank]) -> String {
        let rank_bytes = self.decode_bytes(ranks);
        String::from_utf8_lossy(&rank_bytes).to_string()
    }

    fn find_match(text: &[Tok], rule: &MergeRule) -> Option<usize> {
        let mut text_iter = text.iter().enumerate().peekable();
        while let Some((idx, word)) = text_iter.next() {
            if word == &rule.0
                && let Some((_, next_word)) = text_iter.peek()
                && next_word == &&rule.1
            {
                return Some(idx);
            }
        }
        None
    }

    fn parse_merge_rules(
        vocab_file: impl Read,
        bpe_to_byte: &HashMap<char, u8>,
    ) -> Result<Vec<MergeRule>, Box<dyn Error>> {
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
                let a_bytes = a.chars().map(|c| bpe_to_byte[&c]).collect();
                let b_bytes = b.chars().map(|c| bpe_to_byte[&c]).collect();
                Ok(MergeRule(a_bytes, b_bytes))
            })
            .collect()
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

    /// Makes two checks:
    ///
    /// 1. A round trip of string -> token IDs -> string; should result in the original string.
    /// 2. A comparison against [tiktoken-rs].
    ///
    /// [tiktoken_rs]: https://crates.io/crates/tiktoken-rs
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

    static TOKENIZER: LazyLock<Result<Tokenizer, String>> = LazyLock::new(|| {
        let vocab_bpe = include_str!("test_assets/vocab.bpe");
        let encoder_json = include_str!("test_assets/encodings.txt");
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
}
