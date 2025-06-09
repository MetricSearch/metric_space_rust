use crate::container::BitsContainer;
use deepsize::DeepSizeOf;
use std::hash::Hash;
use std::hash::Hasher;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BitSimd<C, const W: usize> {
    container: C,
}

impl<C: BitsContainer, const W: usize> BitSimd<C, W> {
    fn new(container: C) -> Self {
        Self { container }
    }

    pub fn set_bit(&mut self, index: usize, value: bool) {
        if index >= W {
            panic!("cannot set bit {index} in BitSimd of width {W}");
        }

        self.container.set_bit(index, value);
    }

    pub fn and_cloned(&self, other: &Self) -> Self {
        Self::new(self.container.and_cloned(&other.container))
    }

    pub fn count_ones(&self) -> usize {
        self.container.count_ones()
    }
}

impl<C, const W: usize> DeepSizeOf for BitSimd<C, W> {
    fn deep_size_of_children(&self, _context: &mut deepsize::Context) -> usize {
        size_of::<C>()
    }
}

impl<C: BitsContainer, const W: usize> Hash for BitSimd<C, W> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.container.hash::<_, W>(state);
    }
}
