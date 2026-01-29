use crate::ffn::layer::LayerTransform;
use crate::tensor::{Tensor, Vector};

pub struct Ffn {
    layers: Vec<Layer>,
}

struct Layer {
    neurons: Vector,
    transform: LayerTransform,
}

impl Ffn {
    pub fn new(transforms: Vec<LayerTransform>) -> Self {
        assert_ne!(transforms.len(), 0, "transforms can't be empty");
        Self {
            layers: transforms
                .into_iter()
                .map(|t| Layer {
                    neurons: Tensor::new_vector(t.out_dims()),
                    transform: t,
                })
                .collect(),
        }
    }

    pub fn apply(&mut self, inputs: &Vector) {
        // Apply the first layer manually. Then, every next layer will apply the transform from the layer before it. We
        // have to use indexing to make the lifetimes work.
        let Layer { neurons, transform } = &mut self.layers[0];
        transform.apply(inputs, neurons);
        for i in 1..self.layers.len() {
            let (prev, curr) = self.layers.split_at_mut(i);
            let Layer {
                neurons: prev_neurons, ..
            } = &prev[i - 1];
            let Layer {
                neurons: curr_neurons,
                transform: curr_transform,
            } = &mut curr[0];
            curr_transform.apply(prev_neurons, curr_neurons);
        }
    }

    pub fn get(&self) -> &Vector {
        &self.layers[self.layers.len() - 1].neurons
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::assert_f32_slice;

    /// Compares against a reference pytorch implementation.
    ///
    /// The pytorch FFN uses:
    /// - input dimension 5
    /// - one hidden layer, dimension 7
    /// - output dimension 6
    /// - tanh-approximated activation
    ///
    /// The code is:
    ///
    /// ```python
    /// import torch
    /// import torch.nn as nn
    ///
    /// class FFN(nn.Module):
    /// def __init__(self):
    /// super(FFN, self).__init__()
    /// self.fc1 = nn.Linear(5, 7)
    /// self.gelu = nn.GELU(approximate='tanh')
    /// self.fc2 = nn.Linear(7, 6)
    ///
    /// def forward(self, x):
    /// x = self.fc1(x)
    /// x = self.gelu(x)
    /// x = self.fc2(x)
    /// return x
    ///
    /// model = FFN()
    ///
    /// # Set fc1 weights and biases. Each set of parameters will have values 1-N
    /// with torch.no_grad():
    /// model.fc1.weight = nn.Parameter(torch.arange(1, 36).float().reshape(7, 5))
    /// model.fc1.bias = nn.Parameter(torch.arange(1, 8).float())
    ///
    /// model.fc2.weight = nn.Parameter(torch.arange(1, 43).float().reshape(6, 7))
    /// model.fc2.bias = nn.Parameter(torch.arange(1, 7).float())
    ///
    /// # Fixed input, also 1-N
    /// input_tensor = torch.arange(1, 6).float().unsqueeze(0)  # Shape: [1, 5]; unsqueeze to add batch dimension
    ///
    /// output = model(input_tensor)
    /// print("Output:", output.tolist())
    /// ```
    ///
    /// Output is:
    ///
    /// ```text
    /// Output: [[10081.0, 23998.0, 37915.0, 51832.0, 65749.0, 79666.0]]
    /// ```
    ///
    #[test]
    fn compare_against_pytorch() {
        let mut ffn = Ffn::new(vec![LayerTransform::new(5, 7), LayerTransform::new(7, 6)]);
        for Layer { transform, .. } in &mut ffn.layers {
            let weight_size = transform.in_dims() * transform.out_dims();
            let bias_size = transform.out_dims();
            transform.set_weights(&count_up(weight_size), &count_up(bias_size));
        }

        let mut input = Tensor::new_vector(5);
        input.set_all(&count_up(5));

        ffn.apply(&input);

        let mut expect = Tensor::new_vector(6);
        expect.set_all(&[10081.0, 23998.0, 37915.0, 51832.0, 65749.0, 79666.0]);

        assert_f32_slice!(ffn.get().as_f32(), expect.as_f32());
    }

    fn count_up(num_elems: usize) -> Vec<f32> {
        (0..num_elems).map(|i| (i + 1) as f32).collect()
    }
}
