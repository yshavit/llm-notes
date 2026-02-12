use clap::ValueEnum;

#[derive(ValueEnum, Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub enum Gpt2Size {
    #[value(name = "124M")]
    Size124M,
    #[value(name = "355M")]
    Size355M,
    #[value(name = "774M")]
    Size774M,
    #[value(name = "1558M")]
    Size1558M,
}

impl Gpt2Size {
    pub fn size(self) -> &'static str {
        match self {
            Gpt2Size::Size124M => "124M",
            Gpt2Size::Size355M => "355M",
            Gpt2Size::Size774M => "774M",
            Gpt2Size::Size1558M => "1558M",
        }
    }
}
