use candle_core::{DType, Device, IndexOp};
use candle_nn;
use simpllm::tensor::{Shape, Tensor, TensorBackend};
use std::borrow::Cow;
use std::error::Error;
use std::sync::LazyLock;

fn main() -> Result<(), Box<dyn Error>> {
    simpllm::run::run_main::<CandleBackend>()
}

static CUDA: LazyLock<Device> = LazyLock::new(|| Device::new_cuda(0).expect("couldn't initialize CUDA"));

struct CandleBackend;

impl TensorBackend for CandleBackend {
    type Tensor<const R: usize> = CandleTensor<R>;
}

#[derive(Clone)]
struct CandleTensor<const R: usize> {
    c_tensor: candle_core::Tensor,
}

impl<const R: usize> Tensor<R> for CandleTensor<R> {
    type Backend = CandleBackend;

    fn new(shape: impl Into<Shape<R>>) -> Self {
        let shape: Shape<R> = shape.into();
        let candle_shape: Vec<usize> = shape.iter().copied().collect();
        let c_tensor =
            candle_core::Tensor::zeros(candle_shape, DType::F32, &CUDA).expect("couldn't create candle tensor");
        Self { c_tensor }
    }

    fn shape(&self) -> Shape<R> {
        let dims = self.c_tensor.dims();
        let array: [usize; R] = dims.try_into().expect("shape rank mismatch");
        Shape::from(array)
    }

    fn reset_values(&mut self, values: &[f32]) {
        self.c_tensor = candle_core::Tensor::from_slice(values, self.c_tensor.shape(), self.c_tensor.device())
            .expect("couldn't create new tensor");
    }

    fn reshape<const R2: usize>(self, new_shape: impl Into<Shape<R2>>) -> <Self::Backend as TensorBackend>::Tensor<R2> {
        let shape: Shape<R2> = new_shape.into();
        let candle_shape: Vec<usize> = shape.iter().copied().collect();
        let c_tensor = self.c_tensor.reshape(candle_shape).expect("couldn't reshape tensor");
        CandleTensor { c_tensor }
    }

    fn split<const S: usize>(self, dim: usize) -> [<Self::Backend as TensorBackend>::Tensor<R>; S] {
        let chunk_size = self.c_tensor.dim(dim).expect("invalid dim") / S;
        let chunks = self.c_tensor.chunk(S, dim).expect("couldn't split tensor");
        let array: [candle_core::Tensor; S] = chunks.try_into().expect("split size mismatch");
        array.map(|c_tensor| CandleTensor { c_tensor })
    }

    fn transposed(self, dim0: usize, dim1: usize) -> Self {
        let c_tensor = self.c_tensor.transpose(dim0, dim1).expect("couldn't transpose tensor");
        Self { c_tensor }
    }

    fn contiguous(self) -> Self {
        let c_tensor = self.c_tensor.contiguous().expect("couldn't make tensor contiguous");
        Self { c_tensor }
    }

    fn matmul(&self, other: &Self) -> Self {
        let c_tensor = self.c_tensor.matmul(&other.c_tensor).expect("couldn't perform matmul");
        Self { c_tensor }
    }

    fn get(&self, indices: [usize; R]) -> f32 {
        let r = match R {
            1 => self.c_tensor.i(indices[0]),
            2 => self.c_tensor.i((indices[0], indices[1])),
            3 => self.c_tensor.i((indices[0], indices[1], indices[2])),
            4 => self.c_tensor.i((indices[0], indices[1], indices[2], indices[3])),
            _ => panic!("unsupported"),
        };
        r.unwrap().to_scalar().unwrap()
    }

    fn with_row<X>(&self, indices: [usize; R], f: impl FnOnce(&[f32]) -> X) -> X {
        let mut tensor = self.c_tensor.clone();
        for (i, &idx) in indices.iter().enumerate().take(R - 1) {
            tensor = tensor.narrow(i, idx, 1).expect("couldn't narrow tensor");
        }
        tensor = tensor
            .narrow(R - 1, indices[R - 1], self.shape()[R - 1])
            .expect("couldn't narrow tensor");
        let flat = tensor.flatten_all().expect("couldn't flatten tensor");
        let vec = flat.to_vec1::<f32>().expect("couldn't convert to vec");
        f(&vec)
    }

    fn mut_row<X>(&mut self, indices: [usize; R], f: impl FnOnce(&mut [f32]) -> X) -> X {
        let mut tensor = self.c_tensor.clone();
        for (i, &idx) in indices.iter().enumerate().take(R - 1) {
            tensor = tensor.narrow(i, idx, 1).expect("couldn't narrow tensor");
        }
        tensor = tensor
            .narrow(R - 1, indices[R - 1], self.shape()[R - 1])
            .expect("couldn't narrow tensor");
        let flat = tensor.flatten_all().expect("couldn't flatten tensor");
        let mut vec = flat.to_vec1::<f32>().expect("couldn't convert to vec");
        let result = f(&mut vec);

        // Write the modified values back
        let row_tensor =
            candle_core::Tensor::from_slice(&vec, flat.shape(), flat.device()).expect("couldn't create tensor");
        let mut slice_params = vec![];
        for (i, &idx) in indices.iter().enumerate().take(R - 1) {
            slice_params.push(idx..idx + 1);
        }
        slice_params.push(indices[R - 1]..indices[R - 1] + self.shape()[R - 1]);
        self.c_tensor = self
            .c_tensor
            .slice_assign(&slice_params, &row_tensor)
            .expect("couldn't assign slice");

        result
    }

    fn flat_f32(&self) -> Cow<'_, [f32]> {
        let flat = self.c_tensor.flatten_all().expect("couldn't flatten tensor");
        let vec = flat.to_vec1::<f32>().expect("couldn't convert to vec");
        Cow::Owned(vec)
    }

    fn gelu(self) -> Self {
        let c_tensor = self.c_tensor.gelu().expect("couldn't apply gelu");
        Self { c_tensor }
    }

    fn softmax(self) -> Self {
        let c_tensor = candle_nn::ops::softmax_last_dim(&self.c_tensor).expect("couldn't apply softmax");
        Self { c_tensor }
    }

    fn multiply_scalar(&mut self, factor: f32) {
        self.c_tensor = self.c_tensor.affine(factor.into(), 0.0).unwrap();
    }

    fn add<const R2: usize>(self, other: <Self::Backend as TensorBackend>::Tensor<R2>) -> Self {
        let c_tensor = self
            .c_tensor
            .broadcast_add(&other.c_tensor)
            .expect("couldn't add tensors");
        Self { c_tensor }
    }

    fn set_row(&mut self, indices: [usize; R], values: &[f32]) {
        let shape = self.shape();
        let row_tensor = candle_core::Tensor::from_slice(values, (values.len(),), self.c_tensor.device())
            .expect("couldn't create tensor from slice");

        let mut slice_params = vec![];
        for (i, &idx) in indices.iter().enumerate().take(R - 1) {
            slice_params.push(idx..idx + 1);
        }
        slice_params.push(indices[R - 1]..indices[R - 1] + shape[R - 1]);

        self.c_tensor = self
            .c_tensor
            .slice_assign(&slice_params, &row_tensor)
            .expect("couldn't assign slice");
    }
}
