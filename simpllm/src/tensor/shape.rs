use std::fmt::{Debug, Display, Formatter};

/// The shape of a tensor. For example, a matrix will have a shape like "5x4".
///
/// This is mostly used for helpful debugging messages.
#[derive(Clone, PartialEq, Eq)]
pub struct Shape {
    dimensions: Vec<usize>,
}

impl Shape {
    pub fn new(dimensions: Vec<usize>) -> Self {
        Self { dimensions }
    }
}

impl Debug for Shape {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut dim_iter = self.dimensions.iter();
        if let Some(dim) = dim_iter.next() {
            write!(f, "{dim}")?;
        } else {
            write!(f, "()")?;
            return Ok(());
        }

        while let Some(dim) = dim_iter.next() {
            write!(f, "x{dim}")?;
        }
        Ok(())
    }
}

impl Display for Shape {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(self, f)
    }
}
