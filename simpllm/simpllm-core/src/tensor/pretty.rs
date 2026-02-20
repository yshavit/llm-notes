use crate::tensor::Tensor;
use std::fmt::{Display, Formatter};

pub struct Pretty<'a, const R: usize, T: Tensor<R>> {
    tensor: &'a T,
    col_limit: Option<usize>,
    precision: Option<usize>,
}

impl<'a, const R: usize, T: Tensor<R>> Pretty<'a, R, T> {
    pub fn with_col_limit(mut self, col_limit: Option<usize>) -> Self {
        self.col_limit = col_limit;
        self
    }

    pub fn with_precision(mut self, precision: Option<usize>) -> Self {
        self.precision = precision;
        self
    }
}

impl<'a, const R: usize, T: Tensor<R>> Display for Pretty<'a, R, T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let tensor = self.tensor;
        let shape = tensor.shape();

        let row_starts: Vec<_> = match R {
            1 => {
                vec![[0; R]]
            }
            2 => (0..shape[0])
                .map(|row_idx| {
                    let mut idx = [0; R];
                    idx[0] = row_idx;
                    idx
                })
                .collect(),
            3 => {
                // Rows first, then batches
                let (n_batches, n_rows) = (shape[0], shape[1]);
                let mut indexes = Vec::with_capacity(n_batches * n_rows);
                for row in 0..n_rows {
                    for batch in 0..n_batches {
                        let mut idx = [0; R];
                        idx[0] = batch;
                        idx[1] = row;
                        indexes.push(idx);
                    }
                }
                indexes
            }
            _ => {
                return write!(f, "({} tensor)", shape);
            }
        };

        let flat_offset = |idx: [usize; R]| -> usize {
            let mut flat_idx = 0;
            let mut stride = 1;
            for i in (0..R).rev() {
                flat_idx += stride * idx[i];
                stride *= shape[i];
            }
            flat_idx
        };

        #[derive(Debug)]
        enum Cell {
            Elem(String),
            Ellipsis,
            ColBreak,
            LineBreak,
        }

        let row_major_f32 = tensor.flat_f32();
        let mut row_major_cells: Vec<Cell> = Vec::with_capacity(shape.num_elements());
        for row_start in row_starts {
            let row_offset = flat_offset(row_start);
            // do the columns
            for col in 0..shape[R - 1] {
                if let Some(col_limit) = self.col_limit
                    && col >= col_limit
                {
                    row_major_cells.push(Cell::Ellipsis);
                    break;
                }
                let val = row_major_f32[row_offset + col];
                let val_str = match self.precision {
                    None => format!("{val}"),
                    Some(precision) => format!("{:.p$}", val, p = precision),
                };
                row_major_cells.push(Cell::Elem(val_str));
            }
            // now either a row break or line break
            match R {
                2 => {
                    let (curr_row, last_row) = (row_start[0], shape[0] - 1);
                    if curr_row < last_row {
                        row_major_cells.push(Cell::LineBreak);
                    }
                }
                3 => {
                    let (curr_batch, last_batch) = (row_start[0], shape[0] - 1);
                    if curr_batch < last_batch {
                        row_major_cells.push(Cell::ColBreak);
                    } else {
                        let (curr_row, last_row) = (row_start[1], shape[1] - 1);
                        if curr_row < last_row {
                            row_major_cells.push(Cell::LineBreak);
                        }
                    }
                }
                _ => {}
            }
        }

        let max_cell_len: usize = row_major_cells
            .iter()
            .map(|c| match c {
                Cell::Elem(s) => s.len(),
                _ => 0,
            })
            .max()
            .unwrap_or(0);

        // Now just render it!
        for cell in row_major_cells {
            match cell {
                Cell::Elem(s) => write!(f, "| {:>width$} ", s, width = max_cell_len)?,
                Cell::ColBreak => write!(f, "|    ")?,
                Cell::LineBreak => writeln!(f, "|")?,
                Cell::Ellipsis => write!(f, "| <...> ")?,
            }
        }
        write!(f, "|")
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
                const PRETTY_COLS_MAX: usize = 3;
                Pretty {
                    tensor: self,
                    col_limit: Some(PRETTY_COLS_MAX),
                    precision: None,
                }
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
