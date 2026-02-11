use std::str::FromStr;

#[derive(Debug, Clone, Copy)]
pub enum OptionalNum<T> {
    Off,
    Value(T),
}

impl<T> FromStr for OptionalNum<T>
where
    T: FromStr,
    T::Err: std::fmt::Display,
{
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.eq_ignore_ascii_case("off") {
            Ok(OptionalNum::Off)
        } else {
            s.parse::<T>()
                .map(OptionalNum::Value)
                .map_err(|e| format!("Invalid value: {}", e))
        }
    }
}

impl<T> From<OptionalNum<T>> for Option<T> {
    fn from(value: OptionalNum<T>) -> Self {
        match value {
            OptionalNum::Off => None,
            OptionalNum::Value(v) => Some(v),
        }
    }
}
