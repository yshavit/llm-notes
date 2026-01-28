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
