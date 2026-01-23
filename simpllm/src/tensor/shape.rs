use std::fmt::{Debug, Display, Formatter};

/// The shape of a tensor. For example, a matrix will have a shape like "5x4".
#[derive(Copy, Clone, PartialEq, Eq)]
pub struct Shape<const R: usize> {
    dimensions: [usize; R],
}

impl<const R: usize> Shape<R> {
    pub fn new(dimensions: [usize; R]) -> Self {
        Self { dimensions }
    }

    pub fn dim(&self) -> &[usize; R] {
        &self.dimensions
    }
}

impl<const R: usize> Debug for Shape<R> {
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

impl<const R: usize> Display for Shape<R> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(self, f)
    }
}
