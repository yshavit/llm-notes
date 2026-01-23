use std::fmt::{Debug, Display, Formatter};
use std::ops::Index;

/// The shape of a tensor. For example, a matrix will have a shape like "5x4".
#[derive(Copy, Clone, PartialEq, Eq)]
pub struct Shape<const R: usize> {
    dimensions: [usize; R],
}

impl<const R: usize> Shape<R> {
    pub fn new(dimensions: [usize; R]) -> Self {
        Self { dimensions }
    }

    pub fn num_elements(self) -> usize {
        self.dimensions.iter().product()
    }

    pub fn swapped(self, dim0: usize, dim1: usize) -> Self {
        let mut dims = self.dimensions;
        dims.swap(dim0, dim1);
        Self::from(dims)
    }

    pub fn iter(&self) -> impl Iterator<Item = &usize> {
        self.dimensions.iter()
    }
}

impl<const R: usize> Index<usize> for Shape<R> {
    type Output = usize;

    fn index(&self, index: usize) -> &Self::Output {
        &self.dimensions[index]
    }
}

impl<const R: usize> From<[usize; R]> for Shape<R> {
    fn from(dimensions: [usize; R]) -> Self {
        Self { dimensions }
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
