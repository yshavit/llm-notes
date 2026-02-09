use crate::cputensor::gelu::gelu;
use crate::cputensor::matmul::matmul_batched;
use crate::cputensor::softmax;
use crate::tensor::Shape;
use rayon::prelude::*;
use std::borrow::Cow;
use std::fmt::{Debug, Display, Formatter};

#[derive(Clone)]
pub struct CpuTensor<const R: usize> {
    data: Vec<f32>,
    shape: Shape<R>,
    strides: Shape<R>,
}

pub type CpuVector = CpuTensor<1>;
pub type CpuMatrix = CpuTensor<2>;

impl<const R: usize> CpuTensor<R> {
    pub(super) fn new<S: Into<Shape<R>>>(shape: S) -> Self {
        assert_ne!(R, 0, "0-tensors are not allowed");
        let shape: Shape<R> = shape.into();
        Self {
            data: vec![0.0; shape.num_elements()],
            shape,
            strides: Self::contiguous_strides(shape),
        }
    }

    fn contiguous_strides(shape: Shape<R>) -> Shape<R> {
        let mut strides = [0; R];
        // Work backwards from the last dimension: The last dimension is contiguous by default (stride = 1), and then
        // each dimension back needs to have a stride-size for all the dimensions before it.
        let mut stride = 1;
        for i in (0..R).rev() {
            strides[i] = stride;
            stride *= shape[i];
        }
        strides.into()
    }

    pub(super) fn shape(&self) -> Shape<R> {
        self.shape
    }

    pub(super) fn reset_values(&mut self, values: &[f32]) {
        assert_eq!(self.shape.num_elements(), values.len());
        // reset to be contiguous
        self.strides = Self::contiguous_strides(self.shape);
        self.data.copy_from_slice(values);
    }

    pub(super) fn split<const S: usize>(self, dim: usize) -> [CpuTensor<R>; S] {
        assert!(dim < R, "split dimension {dim} must be < rank {R}");

        let orig_dim_size = self.shape[dim];
        let ideal_dim_size = orig_dim_size / S;
        assert!(ideal_dim_size > 0, "can't split {} into {S}", self.shape);

        let mut split_tensors = {
            let mut shapes = [self.shape; S];
            shapes.iter_mut().for_each(|s| s[dim] = ideal_dim_size);
            // if the last dimension is too big, trim it down
            let dim_extra = orig_dim_size - shapes.iter().map(|s| s[dim]).sum::<usize>();
            shapes[shapes.len() - 1][dim] -= dim_extra;
            shapes.map(CpuTensor::new)
        };
        // Now copy over the slices. We'll make sure we'ere contiguous. Then we'll Iterate over batches of
        // [a, b, ..., <dim>, 0, 0, 0, ...]. For each of those, we'll take the full slice, divide it by S, and then
        // write it to the corresponding tensor data.
        let contiguous_self = self.contiguous();
        let batch_len = {
            let mut len = contiguous_self.data.len();
            for shape_idx in 0..dim {
                len /= contiguous_self.shape[shape_idx]
            }
            len
        };
        for batch in contiguous_self.shape.iter_indices().skipping_dims_at(dim) {
            let batch_start = contiguous_self.data_offset(batch);
            let batch_slice = &contiguous_self.data[batch_start..(batch_start + batch_len)];
            for (slice_idx, slice) in batch_slice.chunks(batch_len / S).enumerate() {
                let target_tensor = &mut split_tensors[slice_idx];
                let target_offset = target_tensor.data_offset(batch);
                target_tensor.data[target_offset..(target_offset + slice.len())].copy_from_slice(slice);
            }
        }
        split_tensors
    }

    fn data_offset(&self, indices: [usize; R]) -> usize {
        let mut offset = 0;
        for i in 0..R {
            assert!(
                indices[i] < self.shape[i],
                "index out of range: can't get {indices:?} on {} tensor",
                self.shape
            );
            offset += self.strides[i] * indices[i]
        }
        offset
    }

    pub(super) fn get(&self, indices: [usize; R]) -> f32 {
        self.data[self.data_offset(indices)]
    }

    /// Performs an action on a row and returns the result.
    ///
    /// The row indices work similarly to [`Self::set_row`].
    ///
    /// The row is provided as a borrowed slice for efficiency: if it's possible to read it directly from the tensor's
    /// underlying data, then this method will do that.
    pub(super) fn mut_row<X>(&mut self, indices: [usize; R], f: impl FnOnce(&mut [f32]) -> X) -> X {
        let read_start = self.data_offset(indices);
        let read_len = self.shape[R - 1] - indices[R - 1];

        if self.strides[R - 1] == 1 {
            // Row-major format; we can just memcpy
            let data = &mut self.data[read_start..read_start + read_len];
            f(data)
        } else {
            let mut data = vec![0.; read_len];
            self.with_row(indices, |row| data.copy_from_slice(row));
            let result = f(&mut data);
            // now, write it back
            self.set_row(indices, &data);
            result
        }
    }

    fn mut_rows_at_batch(&mut self, base_indexes: [usize; R], f: impl Fn(usize, &mut [f32]) + Sync) {
        assert_eq!(base_indexes[R - 1], 0);
        if R == 1 {
            f(0, &mut self.data);
            return;
        }
        if self.strides == Self::contiguous_strides(self.shape) {
            assert_eq!(base_indexes[R - 2], 0);
            let start_idx = self.data_offset(base_indexes);
            let chunk_num_rows = self.shape[R - 2];
            let chunk_num_cols = self.shape[R - 1];
            let slice_len = chunk_num_cols * chunk_num_rows;
            let data_slice = &mut self.data[start_idx..(start_idx + slice_len)];
            data_slice
                .par_chunks_exact_mut(chunk_num_cols)
                .enumerate()
                .for_each(|(idx, chunk)| {
                    assert_eq!(chunk.len(), chunk_num_cols, "internal error in mut_rows");
                    f(idx, chunk);
                })
        } else {
            todo!("transposed mut_rows not supported");
        }
    }

    pub(super) fn multiply_scalar(&mut self, factor: f32) {
        for v in &mut self.data {
            *v = *v * factor;
        }
    }

    pub(super) fn add_tensor<const R2: usize>(mut self, other: &CpuTensor<R2>) -> Self {
        assert!(
            R2 <= R && self.shape[R - R2..] == other.shape[..],
            "can't add broadcasted {} into {}",
            other.shape,
            self.shape
        );

        // Fast path: same rank and strides - simple element-wise addition
        if R == R2 && self.strides.as_ref() == other.strides.as_ref() {
            other.data.iter().enumerate().for_each(|(i, v)| self.data[i] += v);
            return self;
        }

        // Iterate through the "other" matrix's rows
        for other_batch in other.shape.iter_indices().skipping_dims_at(R2 - 1) {
            other.with_row(other_batch, |other_row| {
                // iterate over self's batch indices (the ones to be broadcast against)
                for mut self_batch in self.shape.iter_indices().skipping_dims_at(R - R2) {
                    self_batch[R - R2..].copy_from_slice(&other_batch);

                    self.mut_row(self_batch, |self_row| {
                        self_row.iter_mut().enumerate().for_each(|(j, v)| *v += other_row[j])
                    })
                }
            });
        }
        self
    }

    pub(super) fn matmul_todo(&self, other: &Self) -> Self {
        matmul_batched(self, other)
    }

    pub(super) fn contiguous(mut self) -> Self {
        if self.strides == Self::contiguous_strides(self.shape) {
            return self;
        }
        let mut new_data = vec![0.0; self.shape.num_elements()];
        let chunk_sizes = self.data.len() / self.shape[0];

        let self_shape = self.shape;
        new_data
            .par_chunks_exact_mut(chunk_sizes)
            .enumerate()
            .for_each(|(row_idx, row)| {
                let mut start_at = [0; R];
                start_at[0] = row_idx;
                let mut index_iter = self_shape.iter_indices_starting_at(start_at).enumerate();
                while let Some((data_idx, tensor_idx)) = index_iter.next() {
                    if tensor_idx[0] > row_idx {
                        break;
                    }
                    row[data_idx] = self.get(tensor_idx);
                }
            });

        self.data = new_data;
        self.strides = Self::contiguous_strides(self.shape);
        self
    }

    pub(super) fn flat_f32(&self) -> Cow<'_, [f32]> {
        if self.strides == Self::contiguous_strides(self.shape) {
            Cow::from(&self.data)
        } else {
            Cow::from(self.clone().contiguous().data)
        }
    }

    pub(super) fn gelu(mut self) -> Self {
        self.data.iter_mut().for_each(|x| *x = gelu(*x));
        self
    }

    pub(super) fn softmax(mut self) -> Self {
        for batches in self.shape.iter_indices().skipping_dims_at(R - 1) {
            self.mut_row(batches, softmax)
        }
        self
    }

    pub(super) fn reshape<const R2: usize>(self, new_shape: impl Into<Shape<R2>>) -> CpuTensor<R2> {
        assert_eq!(
            self.strides,
            Self::contiguous_strides(self.shape),
            "can only shape contiguous tensors"
        );
        let new_shape = new_shape.into();
        assert_eq!(
            self.shape.num_elements(),
            new_shape.num_elements(),
            "can't reshape {} into {}",
            self.shape,
            new_shape
        );

        CpuTensor {
            data: self.data,
            shape: new_shape,
            strides: CpuTensor::contiguous_strides(new_shape),
        }
    }

    pub(super) fn with_row<X>(&self, indices: [usize; R], f: impl FnOnce(&[f32]) -> X) -> X {
        let read_start = self.data_offset(indices);
        let read_len = self.shape[R - 1] - indices[R - 1];

        if self.strides[R - 1] == 1 {
            // Row-major format; we can just memcpy
            let data = &self.data[read_start..read_start + read_len];
            f(data)
        } else {
            let stride: usize = (0..R - 1).map(|i| self.strides[i] * self.shape[i]).sum();
            let mut data = vec![0.; read_len];
            let mut offset = read_start;
            for idx in 0..data.len() {
                data[idx] = self.data[offset];
                offset += stride;
            }
            f(&mut data)
        }
    }

    pub(super) fn transposed(mut self, dim0: usize, dim1: usize) -> Self {
        self.shape.swap(dim0, dim1);
        self.strides.swap(dim0, dim1);
        self
    }

    /// Returns a matrix slice of this tensor.
    ///
    /// The tensor must be at least rank 2. The batch is actually `R-2`, with the last two indices of this tensor being
    /// the ones that the [`MatrixView`] will represent. (Rust doesn't let us specify a type of `R-2`.) As such, the
    /// last two elements of `batch` must be `0`.
    pub(super) fn matrix_slice(&self, batch: [usize; R]) -> MatrixView<'_, R> {
        assert!(R >= 2, "cannot take matrix slice on vectors");
        assert!(
            batch[R - 1] == 0 && batch[R - 2] == 0,
            "invalid batch {batch:?} into {} tensor",
            self.shape
        );
        MatrixView::new(self, batch)
    }

    pub(super) fn matrix_slice_mut(&mut self, batch: [usize; R]) -> MatrixViewMut<'_, R> {
        assert!(R >= 2, "cannot take matrix slice on vectors");
        assert!(
            batch[R - 1] == 0 && batch[R - 2] == 0,
            "invalid batch {batch:?} into {} tensor",
            self.shape
        );
        MatrixViewMut::new(self, batch)
    }

    /// Sets all or part of a row's values. The first `R-1` indices specify an offset into the tensor, and the last
    /// index specifies an offset into the row. That offset + `values.len()` must be less than the last index's
    /// dimensionality.
    pub(super) fn set_row(&mut self, indices: [usize; R], values: &[f32]) {
        let write_start = self.data_offset(indices);
        let row_offset = indices[R - 1];
        assert!(
            values.len() <= self.shape[R - 1] - row_offset,
            "can't write to index {:?} of {} matrix with slice length {}",
            indices,
            self.shape,
            values.len()
        );
        let write_len = self.shape[R - 1] - row_offset;
        if self.strides[R - 1] == 1 {
            // Row-major format; we can just memcpy
            self.data[write_start..write_start + write_len].copy_from_slice(values);
        } else {
            let stride: usize = (0..R - 1).map(|i| self.strides[i] * self.shape[i]).sum();
            let mut offset = write_start;
            for val in values {
                self.data[offset] = *val;
                offset += stride;
            }
        }
    }
}

impl<const R: usize> Debug for CpuTensor<R> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ", self.shape)?;
        match R {
            1 => write!(f, "vector"),
            2 => write!(f, "matrix"),
            _ => write!(f, "tensor"),
        }
    }
}

macro_rules! prettier {
    ($r:literal) => {
        impl Display for CpuTensor<$r> {
            fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
                Pretty(self).fmt(f)
            }
        }
    };
}
prettier! {1}
prettier! {2}
prettier! {3}

macro_rules! matrix_view {
    ($name:ident $($mut:ident)?) => {
        pub(super) struct $name<'a, const R: usize> {
            tensor: &'a $($mut)? CpuTensor<R>,
            batch_dimensions: [usize; R],
        }

        impl<'a, const R: usize> $name<'a, R> {
            fn new(tensor: &'a $($mut)? CpuTensor<R>, batch_dimensions: [usize; R]) -> Self {
                Self {
                    tensor,
                    batch_dimensions,
                }
            }

            pub(super) fn shape(&self) -> Shape<2> {
                Shape::new([self.num_rows(), self.num_cols()])
            }

            pub(super) fn num_rows(&self) -> usize {
                if R == 1 {
                    1
                } else {
                    self.tensor.shape[R - 2]
                }
            }

            pub(super) fn num_cols(&self) -> usize {
                self.tensor.shape[R - 1]
            }

            pub(super) fn get(&self, row: usize, col: usize) -> f32 {
                let mut indices = self.batch_dimensions;
                if R > 1 {
                    indices[R - 2] = row;
                }
                indices[R - 1] = col;
                self.tensor.get(indices)
            }
        }

        impl<'a> From<&'a $($mut)? CpuTensor<2>> for $name<'a, 2> {
            fn from(tensor: &'a $($mut)? CpuTensor<2>) -> Self {
                $name {
                    tensor,
                    batch_dimensions: [0; 2],
                }
            }
        }

    };
}
matrix_view! {MatrixView}
matrix_view! {MatrixViewMut mut}

impl<'a, const R: usize> MatrixViewMut<'a, R> {
    pub(super) fn set_row(&mut self, row: usize, values: &[f32]) {
        let mut indices = self.batch_dimensions;
        if R == 1 {
            assert_eq!(row, 0, "vector's row parameter must be 0")
        } else {
            indices[R - 2] = row;
            indices[R - 1] = 0;
        }
        self.tensor.set_row(indices, values);
    }

    pub(super) fn mut_row(&mut self, row: usize, f: impl Fn(&mut [f32])) {
        let mut indices = self.batch_dimensions;
        if R == 1 {
            assert_eq!(row, 0, "vector's row parameter must be 0")
        } else {
            indices[R - 2] = row;
            indices[R - 1] = 0;
        }
        self.tensor.mut_row(indices, f);
    }

    pub(super) fn mut_rows(&mut self, f: impl Fn(usize, &mut [f32]) + Sync) {
        self.tensor.mut_rows_at_batch(self.batch_dimensions, f);
    }
}

struct Pretty<'a, const R: usize>(&'a CpuTensor<R>);

impl<'a, const R: usize> Display for Pretty<'a, R> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let tensor = self.0;
        if R >= 4 {
            return write!(f, "({} tensor)", tensor.shape);
        }
        // Turn all the values into equal-length, padded strings
        let mut cell_strings = Vec::with_capacity(tensor.shape.num_elements());
        let mut max_cell_len = 0;

        let indices: Box<dyn Iterator<Item = [usize; R]>> = match R {
            1 | 2 => Box::new(tensor.shape.iter_indices()),
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
                let (max_batch, max_row, max_col) = (tensor.shape[0], tensor.shape[1], tensor.shape[2]);
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
        for idx in indices {
            let val_string = tensor.get(idx).to_string();
            max_cell_len = max_cell_len.max(val_string.len());
            cell_strings.push(val_string);
        }
        cell_strings
            .iter_mut()
            .for_each(|s| *s = format!("{:>width$}", s, width = max_cell_len));

        // Now write them all. The cell_strings is already in row-major order, so we can just keep track of newlines.
        let line_length = match R {
            1 => tensor.shape[0],
            2 => tensor.shape[1],
            3 => tensor.shape[0] * tensor.shape[2], // each visual row is a batch and a row
            _ => {
                return Ok(()); // shouldn't ever get here!
            }
        };
        let mut batch_length = match R {
            3 => Some((tensor.shape[2], tensor.shape[2])), // column lengths
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

impl CpuTensor<2> {
    pub(super) fn new_matrix(num_rows: usize, num_columns: usize) -> Self {
        Self::new(Shape::new([num_rows, num_columns]))
    }

    fn t(self) -> Self {
        self.transposed(0, 1)
    }

    fn num_rows(&self) -> usize {
        self.shape[0]
    }

    fn num_cols(&self) -> usize {
        self.shape[1]
    }

    fn mut_rows(&mut self, f: impl Fn(usize, &mut [f32]) + Sync) {
        self.mut_rows_at_batch([0, 0], f);
    }

    fn to_f32(&self) -> Vec<Vec<f32>> {
        let mut result = Vec::with_capacity(self.num_rows());
        for row in 0..self.num_rows() {
            let cols: Vec<_> = (0..self.num_cols()).map(|col| self.get([row, col])).collect();
            result.push(cols);
        }
        result
    }
}

impl<const R: usize> PartialEq for CpuTensor<R> {
    fn eq(&self, other: &Self) -> bool {
        if self.shape != other.shape {
            return false;
        }

        // Fast path: same strides means same layout, so we can do just a simple == on the data
        if self.strides == other.strides {
            return self.data == other.data;
        }

        // Slow path: different strides, so we have to compare element by element
        for indices in self.shape.iter_indices() {
            if self.get(indices) != other.get(indices) {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;
    use std::panic;

    mod matrix {
        use super::*;

        /// Smoke test of the various shapes of things.
        #[test]
        fn matrix_data_shape() {
            let m = CpuTensor::new_matrix(3, 4);
            assert_eq!(m.num_rows(), 3);
            assert_eq!(m.num_cols(), 4);
            assert_eq!(m.shape(), Shape::new([3, 4]));

            check_row(&m, 0, [0., 0., 0., 0.]);
            check_row(&m, 1, [0., 0., 0., 0.]);
            check_row(&m, 2, [0., 0., 0., 0.]);
            expect_panic(|| m.get([3, 0]));
        }

        #[test]
        fn data_round_trip() {
            let mut m = CpuTensor::new_matrix(3, 4);

            m.set_row([0, 0], &[1., 2., 3., 4.]);
            m.set_row([1, 0], &[5., 6., 7., 8.]);
            m.set_row([2, 0], &[9., 10., 11., 12.]);

            check_row(&m, 0, [1., 2., 3., 4.]);
            check_row(&m, 1, [5., 6., 7., 8.]);
            check_row(&m, 2, [9., 10., 11., 12.]);

            m.mut_row([1, 0], |data| data.iter_mut().for_each(|v| *v = *v * 10.));
            check_row(&m, 1, [50., 60., 70., 80.]);
            // sub-row
            m.mut_row([1, 1], |data| data.iter_mut().for_each(|v| *v = *v * 10.));
            check_row(&m, 1, [50., 600., 700., 800.]);
        }

        #[test]
        #[should_panic = "can't write to index [0, 0] of 3x4 matrix with slice length 6"]
        fn row_mut_set_all_bounds() {
            let mut m = CpuTensor::new_matrix(3, 4);
            m.set_row([0, 0], &[1., 2., 3., 4., 5., 6.]);
        }

        #[test]
        fn transposition() {
            let mut m = CpuTensor::new_matrix(3, 4);

            m.set_row([0, 0], &[1., 2., 3., 4.]);
            m.set_row([1, 0], &[5., 6., 7., 8.]);
            m.set_row([2, 0], &[9., 10., 11., 12.]);

            let transposed = m.t();
            assert_eq!(transposed.shape(), Shape::new([4, 3]));

            check_row(&transposed, 0, [1., 5., 9.]);
            check_row(&transposed, 1, [2., 6., 10.]);
            check_row(&transposed, 2, [3., 7., 11.]);
            check_row(&transposed, 3, [4., 8., 12.]);
            expect_panic(|| transposed.get([4, 0]));

            // quick sanity check on double-transposition
            let double_transposed = transposed.t();
            check_row(&double_transposed, 0, [1., 2., 3., 4.]);
        }

        #[test]
        fn transposed_set_row() {
            let m = CpuTensor::new_matrix(3, 4);

            let mut transposed = m.t();
            let values = &[1., 2., 3.];
            transposed.set_row([1, 0], values);

            check_row(&transposed, 0, [0., 0., 0.]);
            check_row(&transposed, 1, [1., 2., 3.]);
            check_row(&transposed, 2, [0., 0., 0.]);
            check_row(&transposed, 3, [0., 0., 0.]);

            transposed.mut_row([1, 0], |data| data.iter_mut().for_each(|v| *v = *v * 10.));
            check_row(&transposed, 1, [10., 20., 30.]);
            // sub-row
            transposed.mut_row([1, 1], |data| data.iter_mut().for_each(|v| *v = *v * 10.));
            check_row(&transposed, 1, [10., 200., 300.]);
        }

        fn check_row<const N: usize>(m: &CpuMatrix, row: usize, expected: [f32; N]) {
            let actual: Vec<_> = (0..N).map(|i| m.get([row, i])).collect();
            let expected = Vec::from(expected);
            assert_eq!(actual, expected);

            expect_panic(|| m.get([row, N]))
        }
    }

    mod tensor3 {
        use super::*;

        type Tensor3 = CpuTensor<3>;

        #[test]
        fn shape() {
            let t = Tensor3::new(Shape::new([2, 3, 4]));
            assert_eq!(t.shape(), Shape::new([2, 3, 4]));

            // Spot-check that initial values are 0.0
            assert_eq!(t.get([0, 0, 0]), 0.0);
            assert_eq!(t.get([1, 2, 3]), 0.0);
        }

        #[test]
        fn get_set_round_trip() {
            let mut t = Tensor3::new(Shape::new([2, 3, 4]));

            // Set values in different "slices"
            t.set_row([0, 0, 0], &[1., 2., 3., 4.]);
            t.set_row([0, 1, 0], &[5., 6., 7., 8.]);
            t.set_row([1, 0, 0], &[9., 10., 11., 12.]);
            t.set_row([1, 2, 0], &[13., 14., 15., 16.]);

            // Verify they're in the right places
            assert_eq!(t.get([0, 0, 0]), 1.);
            assert_eq!(t.get([0, 0, 3]), 4.);
            assert_eq!(t.get([0, 1, 1]), 6.);
            assert_eq!(t.get([1, 0, 2]), 11.);
            assert_eq!(t.get([1, 2, 3]), 16.);

            // Values we didn't set should still be 0
            assert_eq!(t.get([0, 2, 0]), 0.);
        }

        #[test]
        fn set_row() {
            let mut t = Tensor3::new(Shape::new([2, 3, 4]));

            t.set_row([0, 1, 0], &[10., 20., 30., 40.]);

            // Check the row we set
            assert_eq!(t.get([0, 1, 0]), 10.);
            assert_eq!(t.get([0, 1, 1]), 20.);
            assert_eq!(t.get([0, 1, 2]), 30.);
            assert_eq!(t.get([0, 1, 3]), 40.);

            // Check that other rows weren't affected
            assert_eq!(t.get([0, 0, 0]), 0.);
            assert_eq!(t.get([0, 2, 0]), 0.);
            assert_eq!(t.get([1, 1, 0]), 0.);
        }

        #[test]
        fn transposition_dims_0_1() {
            let mut t = Tensor3::new(Shape::new([2, 3, 4]));

            // Fill with distinct values
            t.set_row([0, 0, 0], &[1., 2., 3., 4.]);
            t.set_row([0, 1, 0], &[5., 6., 7., 8.]);
            t.set_row([0, 2, 0], &[9., 10., 11., 12.]);
            t.set_row([1, 0, 0], &[13., 14., 15., 16.]);
            t.set_row([1, 1, 0], &[17., 18., 19., 20.]);
            t.set_row([1, 2, 0], &[21., 22., 23., 24.]);

            let transposed = t.transposed(0, 1);
            assert_eq!(transposed.shape(), Shape::new([3, 2, 4]));

            // Original [0, 1, 2] should now be at [1, 0, 2]
            assert_eq!(transposed.get([1, 0, 2]), 7.);
            // Original [1, 0, 3] should now be at [0, 1, 3]
            assert_eq!(transposed.get([0, 1, 3]), 16.);
            // Original [1, 2, 1] should now be at [2, 1, 1]
            assert_eq!(transposed.get([2, 1, 1]), 22.);
        }

        #[test]
        #[should_panic = "index out of range: can't get [2, 3, 4] on 2x3x4 tensor"]
        fn bounds_check() {
            let t = Tensor3::new(Shape::new([2, 3, 4]));
            t.get([2, 3, 4]);
        }
    }

    mod matrix_view {
        use super::*;

        #[test]
        fn matrix_view_round_trip() {
            let mut t = CpuTensor::new([4, 2, 3]);

            // Get the second batch (doesn't really matter which)
            let slice_indices = [1, 0, 0];

            let mut t_mut_matrix = t.matrix_slice_mut(slice_indices);

            assert_eq!(t_mut_matrix.shape(), Shape::new([2, 3]));
            assert_eq!(t_mut_matrix.num_rows(), 2);
            assert_eq!(t_mut_matrix.num_cols(), 3);
            t_mut_matrix.set_row(0, &[1., 2., 3.]);
            t_mut_matrix.set_row(1, &[4., 5., 6.]);

            // just some spot checks
            assert_eq!(t_mut_matrix.get(0, 0), 1.);
            assert_eq!(t_mut_matrix.get(1, 2), 6.);

            // check the non-mut version too. again, just some spot checks
            let t_matrix = t.matrix_slice(slice_indices);
            assert_eq!(t_matrix.get(0, 1), 2.);
            assert_eq!(t_matrix.get(1, 1), 5.);
        }
    }

    mod reshape {
        use super::*;

        #[test]
        fn reshape_2_to_3() {
            // use pretty-print to check values

            let mut original = CpuTensor::new_matrix(2, 6);
            original.set_row([0, 0], &[01., 02., 03., 04., 05., 06.]);
            original.set_row([1, 0], &[07., 08., 09., 10., 11., 12.]);
            assert_eq!(
                format!("{original}"),
                [
                    //
                    "|  1 |  2 |  3 |  4 |  5 |  6 |",
                    "|  7 |  8 |  9 | 10 | 11 | 12 |",
                ]
                .join("\n")
            );

            let reshaped = original.reshape([2, 2, 3]);
            assert_eq!(
                format!("{reshaped}"),
                [
                    //
                    "|  1 |  2 |  3 |    |  7 |  8 |  9 |",
                    "|  4 |  5 |  6 |    | 10 | 11 | 12 |",
                ]
                .join("\n")
            );
            assert_eq!(reshaped.get([0, 0, 0]), 01.);
            assert_eq!(reshaped.get([0, 0, 1]), 02.);
            assert_eq!(reshaped.get([0, 0, 2]), 03.);
            assert_eq!(reshaped.get([0, 1, 0]), 04.);

            let back_to_orig = reshaped.reshape([2, 6]);
            assert_eq!(
                format!("{back_to_orig}"),
                [
                    //
                    "|  1 |  2 |  3 |  4 |  5 |  6 |",
                    "|  7 |  8 |  9 | 10 | 11 | 12 |",
                ]
                .join("\n")
            );

            let as_vector = back_to_orig.reshape([12]);
            assert_eq!(as_vector.get([11]), 12.);
        }

        #[test]
        #[should_panic]
        fn shape_mismatch() {
            let original = CpuTensor::new_matrix(2, 6);
            let _ = original.reshape([2, 7]);
        }

        #[test]
        #[should_panic]
        fn transposed() {
            let original = CpuTensor::new_matrix(2, 6).transposed(0, 1);
            let _ = original.reshape([6, 2]);
        }
    }

    mod split {
        use super::*;

        #[test]
        fn split_on_last_dim() {
            let mut orig = CpuTensor::new([2, 2, 6]);
            orig.data
                .iter_mut()
                .enumerate()
                .for_each(|(count, value)| *value = count as f32);
            // sanity check
            assert_eq!(
                format!("{orig}"),
                [
                    "|  0 |  1 |  2 |  3 |  4 |  5 |    | 12 | 13 | 14 | 15 | 16 | 17 |",
                    "|  6 |  7 |  8 |  9 | 10 | 11 |    | 18 | 19 | 20 | 21 | 22 | 23 |",
                ]
                .join("\n")
            );

            let [split_1, split_2, split_3] = orig.split::<3>(2);
            assert_eq!(
                format!("{}", split_1),
                [
                    // split 1
                    "|  0 |  1 |    | 12 | 13 |",
                    "|  6 |  7 |    | 18 | 19 |",
                ]
                .join("\n")
            );
            assert_eq!(
                format!("{split_2}"),
                ["|  2 |  3 |    | 14 | 15 |", "|  8 |  9 |    | 20 | 21 |",].join("\n")
            );
            assert_eq!(
                format!("{split_3}"),
                ["|  4 |  5 |    | 16 | 17 |", "| 10 | 11 |    | 22 | 23 |",].join("\n")
            );
        }

        #[test]
        fn split_on_middle_dim() {
            let mut orig = CpuTensor::new([2, 6, 2]);
            orig.data
                .iter_mut()
                .enumerate()
                .for_each(|(count, value)| *value = count as f32);
            // sanity check
            assert_eq!(
                format!("{orig}"),
                [
                    "|  0 |  1 |    | 12 | 13 |",
                    "|  2 |  3 |    | 14 | 15 |",
                    "|  4 |  5 |    | 16 | 17 |",
                    "|  6 |  7 |    | 18 | 19 |",
                    "|  8 |  9 |    | 20 | 21 |",
                    "| 10 | 11 |    | 22 | 23 |",
                ]
                .join("\n")
            );

            let [split_1, split_2, split_3] = orig.split::<3>(1);
            assert_eq!(
                format!("{}", split_1),
                [
                    // split 1 - rows 0-1
                    "|  0 |  1 |    | 12 | 13 |",
                    "|  2 |  3 |    | 14 | 15 |",
                ]
                .join("\n")
            );
            assert_eq!(
                format!("{split_2}"),
                [
                    // split 2 - rows 2-3
                    "|  4 |  5 |    | 16 | 17 |",
                    "|  6 |  7 |    | 18 | 19 |",
                ]
                .join("\n")
            );
            assert_eq!(
                format!("{split_3}"),
                [
                    // split 3 - rows 4-5
                    "|  8 |  9 |    | 20 | 21 |",
                    "| 10 | 11 |    | 22 | 23 |",
                ]
                .join("\n")
            );
        }

        #[test]
        fn split_on_first_dim() {
            let mut orig = CpuTensor::new([6, 2, 2]);
            orig.data
                .iter_mut()
                .enumerate()
                .for_each(|(count, value)| *value = count as f32);
            // sanity check
            assert_eq!(
                format!("{orig}"),
                [
                    "|  0 |  1 |    |  4 |  5 |    |  8 |  9 |    | 12 | 13 |    | 16 | 17 |    | 20 | 21 |",
                    "|  2 |  3 |    |  6 |  7 |    | 10 | 11 |    | 14 | 15 |    | 18 | 19 |    | 22 | 23 |",
                ]
                .join("\n")
            );

            let [split_1, split_2, split_3] = orig.split::<3>(0);
            assert_eq!(
                format!("{}", split_1),
                [
                    // split 1 - first 2 matrices
                    "| 0 | 1 |    | 4 | 5 |",
                    "| 2 | 3 |    | 6 | 7 |",
                ]
                .join("\n")
            );
            assert_eq!(
                format!("{split_2}"),
                [
                    // split 2 - next 2 matrices
                    "|  8 |  9 |    | 12 | 13 |",
                    "| 10 | 11 |    | 14 | 15 |",
                ]
                .join("\n")
            );
            assert_eq!(
                format!("{split_3}"),
                [
                    // split 3 - last 2 matrices
                    "| 16 | 17 |    | 20 | 21 |",
                    "| 18 | 19 |    | 22 | 23 |",
                ]
                .join("\n")
            );
        }
    }

    mod add {
        use super::*;

        #[test]
        fn same_shape() {
            let mut a = CpuTensor::new_matrix(3, 3);
            let mut b = CpuTensor::new_matrix(3, 3);
            a.data.iter_mut().enumerate().for_each(|(n, v)| *v = n as f32);
            b.data.iter_mut().enumerate().for_each(|(n, v)| *v = (n * 10) as f32);

            let c = a.add(&b);

            assert_eq!(
                format!("{c}"),
                [
                    //
                    "|  0 | 11 | 22 |",
                    "| 33 | 44 | 55 |",
                    "| 66 | 77 | 88 |",
                ]
                .join("\n")
            );
        }

        #[test]
        fn broadcast() {
            let mut a = CpuTensor::new([2, 3, 3]);
            let mut b = CpuTensor::new_matrix(3, 3);
            a.data.iter_mut().enumerate().for_each(|(n, v)| *v = (n * 100) as f32);
            b.data.iter_mut().enumerate().for_each(|(n, v)| *v = n as f32);

            assert_eq!(
                format!("{a}"),
                [
                    //
                    "|    0 |  100 |  200 |    |  900 | 1000 | 1100 |",
                    "|  300 |  400 |  500 |    | 1200 | 1300 | 1400 |",
                    "|  600 |  700 |  800 |    | 1500 | 1600 | 1700 |",
                ]
                .join("\n")
            );
            assert_eq!(
                format!("{b}"),
                [
                    //
                    "| 0 | 1 | 2 |",
                    "| 3 | 4 | 5 |",
                    "| 6 | 7 | 8 |",
                ]
                .join("\n")
            );
            let c = a.add(&b);

            assert_eq!(
                format!("{c}"),
                [
                    //
                    "|    0 |  101 |  202 |    |  900 | 1001 | 1102 |",
                    "|  303 |  404 |  505 |    | 1203 | 1304 | 1405 |",
                    "|  606 |  707 |  808 |    | 1506 | 1607 | 1708 |",
                ]
                .join("\n")
            );
        }
    }

    mod pretty {
        use super::*;

        #[test]
        fn pretty_vector() {
            let mut m = CpuTensor::new([4]);

            m.set_row([0], &[1., 2., 3., 4.]);

            let pretty = format!("{m}");
            assert_eq!(pretty, "| 1 | 2 | 3 | 4 |");
        }

        #[test]
        fn pretty_matrix() {
            let mut m = CpuTensor::new_matrix(3, 4);

            m.set_row([0, 0], &[1., 2., 3., 4.]);
            m.set_row([1, 0], &[5., 6., 7., 8.]);
            m.set_row([2, 0], &[9., 10., 11., 12.]);

            let pretty = format!("{m}");

            assert_eq!(
                pretty,
                [
                    //
                    "|  1 |  2 |  3 |  4 |",
                    "|  5 |  6 |  7 |  8 |",
                    "|  9 | 10 | 11 | 12 |",
                ]
                .join("\n")
            );
        }

        #[test]
        fn pretty_tensor_3() {
            let mut m = CpuTensor::new([3, 4, 2]);

            m.set_row([0, 0, 0], &[1., 2.]);
            m.set_row([1, 0, 0], &[3., 4.]);
            m.set_row([2, 0, 0], &[5., 6.]);

            m.set_row([0, 1, 0], &[7., 8.]);
            m.set_row([1, 1, 0], &[9., 10.]);
            m.set_row([2, 1, 0], &[11., 12.]);

            m.set_row([0, 2, 0], &[13., 14.]);
            m.set_row([1, 2, 0], &[15., 16.]);
            m.set_row([2, 2, 0], &[17., 18.]);

            m.set_row([0, 3, 0], &[19., 20.]);
            m.set_row([1, 3, 0], &[21., 22.]);
            m.set_row([2, 3, 0], &[23., 24.]);

            let pretty = format!("{m}");

            assert_eq!(
                pretty,
                [
                    //
                    "|  1 |  2 |    |  3 |  4 |    |  5 |  6 |",
                    "|  7 |  8 |    |  9 | 10 |    | 11 | 12 |",
                    "| 13 | 14 |    | 15 | 16 |    | 17 | 18 |",
                    "| 19 | 20 |    | 21 | 22 |    | 23 | 24 |",
                ]
                .join("\n")
            );
        }

        /// Check the fence post when there's just one batch
        #[test]
        fn pretty_tensor_3_batch_is_1() {
            let mut m = CpuTensor::new([1, 2, 2]);

            m.set_row([0, 0, 0], &[1., 2.]);
            m.set_row([0, 1, 0], &[3., 4.]);
            let pretty = format!("{m}");

            assert_eq!(
                pretty,
                [
                    //
                    "| 1 | 2 |",
                    "| 3 | 4 |",
                ]
                .join("\n")
            );
        }

        #[test]
        fn pretty_tensor_4() {
            let m = CpuTensor::new([1, 2, 3, 4]);

            // can't pretty it via a tensor method; use prettier directly
            let pretty = format!("{}", Pretty(&m));

            // no pretty text for it (which is why we don't expose it)
            assert_eq!(pretty, "(1x2x3x4 tensor)");
        }
    }

    fn expect_panic<X>(f: impl FnOnce() -> X + panic::UnwindSafe) {
        assert!(panic::catch_unwind(f).is_err());
    }
}
