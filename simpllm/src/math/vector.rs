pub trait Vector {
    fn len(&self) -> usize;
    fn iter(&self) -> impl Iterator<Item = f32>;
}

pub trait VectorMut: Vector {
    fn set(&mut self, idx: usize, value: f32);

    fn set_all(&mut self, value: &[f32]) {
        if self.len() != value.len() {
            panic!(
                "can't set {} values into a vector of length {}",
                value.len(),
                self.len()
            );
        }
        for idx in 0..self.len() {
            self.set(idx, value[idx]);
        }
    }
}
