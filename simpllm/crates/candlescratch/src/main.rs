use candle_core::{Device, Tensor, Var};

fn main() -> Result<(), candle_core::Error> {
    let device = Device::Cpu;

    // Input: 1x3
    let x = Tensor::new(
        &[
            //
            [1.0f32, 2.0, 3.0],
        ],
        &device,
    )?;

    // Weight matrix: 3x2
    let w = Var::new(
        &[
            //
            [0.1f32, 0.2],
            [0.3, 0.4],
            [0.5, 0.6],
        ],
        &device,
    )?;

    // Forward pass: y = x @ w  (shape: 1x2)
    let y = x.matmul(&w)?;

    // Simple scalar loss: sum of all outputs
    let loss = y.sum_all()?;
    // This wouldn't work, because we need y to actually tie back to the computational graph
    // let loss = Tensor::new(5.0f32, &device)?;

    println!("loss: {:?}", loss.to_vec0::<f32>()?);

    // Backward pass
    let grads = loss.backward()?;
    println!("grads: {:?}", grads);

    // Get gradient w.r.t. w
    let grad_w = grads.get(&w).unwrap();
    println!("grad_w: {:?}", grad_w.to_vec2::<f32>()?);

    Ok(())
}
