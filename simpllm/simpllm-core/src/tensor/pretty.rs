use crate::tensor::Tensor;
use std::fmt::{Display, Formatter};

pub struct Pretty<'a, const R: usize, T: Tensor<R>>(&'a T);

impl<'a, const R: usize, T: Tensor<R>> Display for Pretty<'a, R, T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let tensor = self.0;
        let shape = tensor.shape();
        if R >= 4 {
            return write!(f, "({} tensor)", shape);
        }
        // Turn all the values into equal-length, padded strings
        let mut cell_strings = Vec::with_capacity(shape.num_elements());
        let mut max_cell_len = 0;

        let indices: Box<dyn Iterator<Item = [usize; R]>> = match R {
            1 | 2 => Box::new(shape.iter_indices()),
            3 => {
                // Different from the standard: each visual row is a batch and then a column.
                //
                // Conceptually:
                // for row in rows:
                //   for batch in batches:
                //      for column in columns:
                //          yield (batch, row, column)
                let mut batch: usize = 0;
                let mut col: usize = 0;
                let mut row: usize = 0;
                let (max_batch, max_row, max_col) = (shape[0], shape[1], shape[2]);
                Box::new(std::iter::from_fn(move || {
                    if col >= max_col {
                        col = 0;
                        batch += 1;
                    }
                    if batch >= max_batch {
                        batch = 0;
                        row += 1;
                    }
                    if row >= max_row {
                        None
                    } else {
                        let mut idx = [0; R];
                        idx.copy_from_slice(&[batch, row, col]);
                        col += 1;
                        Some(idx)
                    }
                }))
            }

            _ => Box::new(std::iter::empty()),
        };

        let row_major_f32 = tensor.flat_f32();
        for idx in indices {
            let flat_idx = {
                let mut flat_idx = 0;
                let mut stride = 1;
                for i in (0..R).rev() {
                    flat_idx += stride * idx[i];
                    stride *= shape[i];
                }
                flat_idx
            };
            let val_string = row_major_f32[flat_idx].to_string();
            max_cell_len = max_cell_len.max(val_string.len());
            cell_strings.push(val_string);
        }
        cell_strings
            .iter_mut()
            .for_each(|s| *s = format!("{:>width$}", s, width = max_cell_len));

        // Now write them all. The cell_strings is already in row-major order, so we can just keep track of newlines.
        let line_length = match R {
            1 => shape[0],
            2 => shape[1],
            3 => shape[0] * shape[2], // each visual row is a batch and a row
            _ => {
                return Ok(()); // shouldn't ever get here!
            }
        };
        let mut batch_length = match R {
            3 => Some((shape[2], shape[2])), // column lengths
            _ => None,
        };
        let mut line_tracker = line_length; // start a newline right away
        let mut first_line = true;
        for s in cell_strings {
            if line_tracker >= line_length {
                if first_line {
                    first_line = false;
                    write!(f, "|")?;
                } else {
                    write!(f, "\n|")?;
                }
                line_tracker = 0;
            }
            if let Some((batch_tracker, batch_length)) = &mut batch_length {
                if batch_tracker >= batch_length {
                    if line_tracker > 0 {
                        write!(f, "    |")?;
                    }
                    *batch_tracker = 0;
                }
                *batch_tracker += 1;
            }
            write!(f, " {s} |")?;
            line_tracker += 1;
        }

        Ok(())
    }
}

pub trait PrettyTensor<const R: usize> {
    type T: Tensor<R>;
    fn pretty(&self) -> Pretty<'_, R, Self::T>;
}

macro_rules! prettier {
    ($r:literal) => {
        impl<T: Tensor<$r>> PrettyTensor<$r> for T {
            type T = T;

            fn pretty(&self) -> Pretty<'_, $r, Self::T> {
                Pretty(self)
            }
        }
    };
}
prettier! {1}
prettier! {2}
prettier! {3}

#[macro_export]
macro_rules! pretty_tensor {
    ($t:ident) => {
        impl Display for $t<1> {
            fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
                use $crate::tensor::PrettyTensor;
                self.pretty().fmt(f)
            }
        }
        impl Display for $t<2> {
            fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
                use $crate::tensor::PrettyTensor;
                self.pretty().fmt(f)
            }
        }
        impl Display for $t<3> {
            fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
                use $crate::tensor::PrettyTensor;
                self.pretty().fmt(f)
            }
        }
    };
}
