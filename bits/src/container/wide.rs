use crate::container::BitsContainer;
use std::hash::{Hash, Hasher};
use wide::u64x4;

impl BitsContainer for [u64x4; 2] {
    fn count_ones(&self) -> usize {
        self.iter()
            .flat_map(|a| a.as_array_ref())
            .map(|e| e.count_ones() as usize)
            .sum()
    }

    fn and_cloned(&self, other: &Self) -> Self {
        [self[0] & other[0], self[1] & other[1]]
    }

    fn set_bit(&mut self, index: usize, value: bool) {
        let element_width = size_of::<u64x4>() * 8;
        let inner_width = size_of::<u64>() * 8;

        let element_index = index / element_width;
        let inner_index = (index % element_width) / inner_width;
        let bit_index = index % inner_width;

        let element = &mut self[element_index];
        let inner = &mut element.as_array_mut()[inner_index];

        if value {
            *inner |= 1 << bit_index;
        } else {
            *inner &= !(1 << bit_index);
        }
    }

    fn hash<H: Hasher, const W: usize>(&self, state: &mut H) {
        self.iter()
            .flat_map(|e| e.to_array())
            .for_each(|e| e.hash(state));

        // todo: only hash first W bits of self
    }
}
