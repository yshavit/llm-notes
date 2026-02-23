use std::collections::HashMap;

/// Adapted from <https://github.com/openai/gpt-2/blob/master/src/encoder.py>.
pub(super) fn gpt2_bpe_char_to_byte() -> HashMap<char, u8> {
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
    let b_to_ch = result.map(|opt| opt.expect("bytes_to_unicode should map all 256 bytes"));

    // Convert that to a char -> Vec<u8>
    b_to_ch.into_iter().enumerate().map(|(b, c)| (c, b as u8)).collect()
}
