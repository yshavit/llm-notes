use std::fmt::{Debug, Display, Formatter};
use std::ops::{Deref, DerefMut};

/// The shape of a tensor. For example, a matrix will have a shape like "5x4".
#[derive(Copy, Clone, PartialEq, Eq)]
pub struct Shape<const R: usize>([usize; R]);

impl<const R: usize> Shape<R> {
    pub fn iter_indices(self) -> impl Iterator<Item = [usize; R]> {
        IndicesIter::new(self)
    }
}

impl<const R: usize> Deref for Shape<R> {
    type Target = [usize; R];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<const R: usize> DerefMut for Shape<R> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<const R: usize> Shape<R> {
    pub fn new(dimensions: [usize; R]) -> Self {
        let result = Self(dimensions);
        if R == 0 || result.iter().any(|i| *i == 0) {
            panic!("illegal shape: {result}");
        }
        result
    }

    pub fn num_elements(self) -> usize {
        self.iter().product()
    }
}

impl<const R: usize> Debug for Shape<R> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut dim_iter = self.iter();
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

struct IndicesIter<const R: usize> {
    shape: Shape<R>,
    next_index: Option<[usize; R]>,
}

impl<const R: usize> IndicesIter<R> {
    fn new(shape: Shape<R>) -> Self {
        Self {
            shape,
            next_index: Some([0; R]),
        }
    }
}

impl<const R: usize> Iterator for IndicesIter<R> {
    type Item = [usize; R];

    fn next(&mut self) -> Option<Self::Item> {
        let result = self.next_index;
        // Now we'll increment for the next one, if there is one
        if let Some(mut next) = self.next_index {
            // We'll through indices in reverse order. For each one, if we can increment it, then we're done and we
            // can mark the increment as successful. Otherwise, we'll set it to 0 and then try the next index.
            // At the end, if we didn't increment any indices, then the whole iterator is done.
            let mut could_increment = false;
            for idx in (0..R).rev() {
                if next[idx] < (self.shape[idx] - 1) {
                    next[idx] += 1;
                    could_increment = true;
                    break;
                } else {
                    next[idx] = 0;
                }
            }
            self.next_index = if could_increment { Some(next) } else { None };
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic]
    fn zero_dimensional() {
        Shape::new([]);
    }

    #[test]
    #[should_panic]
    fn dimension_is_zero() {
        Shape::new([3, 2, 0, 1]);
    }

    mod iter_indices {
        use super::*;

        #[test]
        fn shape_1_max_1() {
            check_indices(Shape::new([1]), vec![[0]]);
        }

        #[test]
        fn shape_1_max_3() {
            check_indices(Shape::new([3]), vec![[0], [1], [2]]);
        }

        #[test]
        fn shape_2() {
            check_indices(
                Shape::new([2, 2]),
                vec![
                    //
                    [0, 0],
                    [0, 1],
                    [1, 0],
                    [1, 1],
                ],
            );
        }

        #[test]
        fn shape_3() {
            check_indices(
                Shape::new([1, 3, 2]),
                vec![
                    //
                    [0, 0, 0],
                    [0, 0, 1],
                    [0, 1, 0],
                    [0, 1, 1],
                    [0, 2, 0],
                    [0, 2, 1],
                ],
            );
        }

        fn check_indices<const R: usize>(shape: Shape<R>, expect: Vec<[usize; R]>) {
            let actual: Vec<_> = shape.iter_indices().collect();
            assert_eq!(actual, expect);
        }
    }
}
