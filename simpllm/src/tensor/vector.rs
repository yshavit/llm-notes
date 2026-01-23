pub trait Vector {
    fn len(&self) -> usize;
    fn get(&self, idx: usize) -> f32;
    fn iter(&self) -> impl Iterator<Item = f32> + '_ {
        (0..self.len()).map(move |i| self.get(i))
    }
}

pub trait VectorMut {
    fn set_all(&mut self, value: &[f32]);
}
