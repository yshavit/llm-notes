use candle_core::{DType, Device, Module};
use candle_nn;
use simpllm_core::tensor::{LayerNorm, Matrix, Shape, Tensor, TensorBackend, Vector};
use std::borrow::Cow;
use std::error::Error;
use std::ops::Deref;
use std::sync::LazyLock;

pub fn main() -> Result<(), Box<dyn Error>> {
    let _ = CUDA.deref(); // force initialization before loading; this just makes the loading eprintln's show up first.
    simpllm::run_main::<CandleBackend>()
}

static CUDA: LazyLock<Device> = LazyLock::new(|| {
    if candle_core::utils::cuda_is_available() {
        Device::new_cuda(0)
            .map(|d| {
                eprintln!("CUDA initialized");
                d
            })
            .unwrap_or_else(|err| {
                eprintln!("Error initializing CUDA: {err}");
                eprintln!("Will use CPU instead.");
                Device::Cpu
            })
    } else if candle_core::utils::metal_is_available() {
        Device::new_metal(0)
            .map(|d| {
                eprintln!("Metal initialized");
                d
            })
            .unwrap_or_else(|err| {
                eprintln!("Error initializing Metal: {err}");
                eprintln!("Will use CPU instead.");
                Device::Cpu
            })
    } else {
        eprintln!("!! Attention !! Neither CUDA nor Metal were available. Will use CPU processing.");
        #[cfg(any(target_os = "windows", target_os = "linux"))]
        eprintln!("Make sure to build with --features cuda");
        #[cfg(target_os = "macos")]
        eprintln!("Make sure to build with --features metal");
        Device::Cpu
    }
});

struct CandleBackend;

struct CandleLayerNorm(candle_nn::LayerNorm);

impl TensorBackend for CandleBackend {
    type Tensor<const R: usize> = CandleTensor<R>;
    type LayerNorm = CandleLayerNorm;

    fn lower_triangle(n: usize) -> Self::Tensor<2> {
        // a row of zeros, and a row of negative infinities
        let zeros_row = candle_core::Tensor::zeros((n, n), DType::F32, &CUDA).unwrap();
        let neg_inf_row = zeros_row.affine(0.0, f64::NEG_INFINITY).unwrap();

        // triangle of 0s on top, 1s on bottom
        let ones_triangle = candle_core::Tensor::tril2(n, DType::U8, &CUDA).unwrap();

        // use where_cond to 0s on the bottom (== 1, the true condition) and -inf on the top
        let mask = ones_triangle.where_cond(&zeros_row, &neg_inf_row).unwrap();

        CandleTensor { t: mask }
    }
}

impl LayerNorm for CandleLayerNorm {
    type B = CandleBackend;

    fn new(scale: Vector<Self::B>, bias: Vector<Self::B>, epsilon: f32) -> Self {
        let scale = scale.t.to_device(&Device::Cpu).unwrap();
        let bias = bias.t.to_device(&Device::Cpu).unwrap();
        Self(candle_nn::LayerNorm::new(scale, bias, epsilon.into()))
    }

    fn apply(&self, input: &Matrix<Self::B>) -> Matrix<Self::B> {
        let input_on_cpu = input.t.to_device(&Device::Cpu).unwrap();
        let mut result = self.0.forward(&input_on_cpu).unwrap();
        result = result.to_device(&CUDA).unwrap();
        CandleTensor { t: result }
    }
}

#[derive(Clone, Debug)]
struct CandleTensor<const R: usize> {
    t: candle_core::Tensor,
}

impl<const R: usize> Tensor<R> for CandleTensor<R> {
    type Backend = CandleBackend;
    type Slice = candle_core::Tensor;

    fn new(shape: impl Into<Shape<R>>) -> Self {
        let candle_shape: Vec<usize> = shape.into().to_vec();
        Self {
            t: candle_core::Tensor::zeros(candle_shape, DType::F32, &CUDA).unwrap(),
        }
    }

    fn shape(&self) -> Shape<R> {
        Shape::new(self.t.dims().try_into().unwrap())
    }

    fn reset_values(&mut self, values: &[f32]) {
        self.t = candle_core::Tensor::from_slice(values, self.t.shape(), self.t.device()).unwrap();
    }

    fn reshape<const R2: usize>(self, new_shape: impl Into<Shape<R2>>) -> <Self::Backend as TensorBackend>::Tensor<R2> {
        let candle_shape: Vec<usize> = new_shape.into().to_vec();
        CandleTensor {
            t: self.t.reshape(candle_shape).unwrap(),
        }
    }

    fn split<const S: usize>(self, dim: usize) -> [<Self::Backend as TensorBackend>::Tensor<R>; S] {
        let chunks = self.t.chunk(S, dim).unwrap();
        let array: [candle_core::Tensor; S] = chunks.try_into().unwrap();
        array.map(|c_tensor| CandleTensor { t: c_tensor })
    }

    fn transposed(self, dim0: usize, dim1: usize) -> Self {
        Self {
            t: self.t.transpose(dim0, dim1).unwrap(),
        }
    }

    fn contiguous(self) -> Self {
        Self {
            t: self.t.contiguous().unwrap(),
        }
    }

    fn matmul(&self, other: &Self) -> Self {
        let lhs = if self.t.is_contiguous() {
            self.t.clone()
        } else {
            self.t.contiguous().unwrap()
        };
        let rhs = if other.t.is_contiguous() {
            other.t.clone()
        } else {
            other.t.contiguous().unwrap()
        };
        Self {
            t: lhs.matmul(&rhs).unwrap(),
        }
    }

    fn with_row<X>(&self, indices: [usize; R], f: impl FnOnce(&[f32]) -> X) -> X {
        let mut tensor = self.t.clone();
        for (i, &idx) in indices.iter().enumerate().take(R - 1) {
            tensor = tensor.narrow(i, idx, 1).unwrap();
        }
        tensor = tensor.narrow(R - 1, indices[R - 1], self.shape()[R - 1]).unwrap();
        let flat = tensor.flatten_all().unwrap();
        let vec = flat.to_vec1::<f32>().unwrap();
        f(&vec)
    }

    fn slice_row<X>(&self, indices: [usize; R], f: impl FnOnce(&Self::Slice) -> X) -> X {
        let mut tensor = self.t.clone();
        for (i, &idx) in indices.iter().enumerate().take(R - 1) {
            tensor = tensor.narrow(i, idx, 1).unwrap();
        }
        tensor = tensor.narrow(R - 1, indices[R - 1], self.shape()[R - 1]).unwrap();
        let flat = tensor.flatten_all().unwrap();
        f(&flat)
    }

    fn set_slice(&mut self, indices: [usize; R], values: &Self::Slice) {
        let mut slice_params = vec![];
        let v_dims = values.dims();
        let mut target_shape = vec![];

        // The input `values` is expected to represent a slice that starts at `indices`.
        // Its dimensions should correspond to the sizes of the ranges we are assigning to.
        // If `values` has fewer dimensions than R, it's likely a flattened or reduced-rank
        // representation of the slice (e.g., a row from a matrix), so we must reshape it
        // to match the destination's rank for `slice_assign`.

        for (i, &idx) in indices.iter().enumerate() {
            // We assume values provides dimensions for the trailing axes.
            // If it's a "row" slice, it might only have 1 dimension (the last one).
            // So we align v_dims to the END of indices.
            let v_dim_idx = i as i32 + (v_dims.len() as i32 - R as i32);
            let size = if v_dim_idx >= 0 { v_dims[v_dim_idx as usize] } else { 1 };
            slice_params.push(idx..idx + size);
            target_shape.push(size);
        }

        let values = if v_dims.len() != R {
            values.reshape(target_shape).unwrap()
        } else {
            values.clone()
        };

        self.t = self.t.slice_assign(&slice_params, &values).unwrap();
    }

    fn extract_row<X>(self, indices: [usize; R], f: impl FnOnce(&mut [f32]) -> X) -> X {
        let mut tensor = self.t.clone();
        for (i, &idx) in indices.iter().enumerate().take(R - 1) {
            tensor = tensor.narrow(i, idx, 1).unwrap();
        }
        tensor = tensor.narrow(R - 1, indices[R - 1], self.shape()[R - 1]).unwrap();
        let flat = tensor.flatten_all().unwrap();
        let mut vec = flat.to_vec1::<f32>().unwrap();
        f(&mut vec)
    }

    fn flat_f32(&self) -> Cow<'_, [f32]> {
        let vec = self.t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        Cow::Owned(vec)
    }

    fn gelu(self) -> Self {
        Self {
            t: self.t.gelu().unwrap(),
        }
    }

    fn softmax(self) -> Self {
        let cpu_tensor = self.t.to_device(&Device::Cpu).unwrap();
        let softmax_tensor = candle_nn::ops::softmax_last_dim(&cpu_tensor).unwrap();
        Self {
            t: softmax_tensor.to_device(self.t.device()).unwrap(),
        }
    }

    fn multiply_scalar(&mut self, factor: f32) {
        self.t = self.t.affine(factor.into(), 0.0).unwrap();
    }

    fn add<const R2: usize>(self, other: &<Self::Backend as TensorBackend>::Tensor<R2>) -> Self {
        Self {
            t: self.t.broadcast_add(&other.t).unwrap(),
        }
    }

    fn set_row(&mut self, indices: [usize; R], values: &[f32]) {
        let shape = self.shape();
        let mut row_shape = vec![1; R - 1];
        row_shape.push(shape[R - 1]);
        let row_tensor = candle_core::Tensor::from_slice(values, row_shape, self.t.device()).unwrap();

        let mut slice_params = vec![];
        for &idx in indices.iter().take(R - 1) {
            slice_params.push(idx..idx + 1);
        }
        slice_params.push(indices[R - 1]..indices[R - 1] + shape[R - 1]);

        self.t = self.t.slice_assign(&slice_params, &row_tensor).unwrap();
    }
}
