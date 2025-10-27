use std::fmt::Debug;
use std::hash::Hash;
use std::hash::Hasher;

pub use wide::{
    simd128::Simd128, simd256::Simd256, simd256p128::Simd256p128, simd256x2::Simd256x2,
    simd256x4::Simd256x4,
};

mod stdsimd;
mod wide;

pub trait BitsContainer: Clone + Send + Sync + Debug {
    fn new() -> Self;

    fn count_ones(&self) -> usize;
    fn and_cloned(&self, other: &Self) -> Self;
    fn xor(&self, other: &Self) -> Self;

    fn set_bit(&mut self, index: usize, value: bool);

    /// Hash with width (W) limit
    // also avoids "cannot impl foreign trait (Hash) for foreign type" issue
    fn hash<H: Hasher, const W: usize>(&self, state: &mut H) {
        self.into_u64_iter()
            .take(W / 64)
            .for_each(|e| e.hash(state));
    }

    fn into_u64_iter(&self) -> impl Iterator<Item = u64>;

    // Gets the indices of the set bits within the container
    fn get_bits_indices(&self) -> impl Iterator<Item = usize> {
        self.into_u64_iter()
            .enumerate()
            // for each u64...
            .flat_map(|(elem_index, value)| {
                // iterate over its bits
                iterate_bits(value)
                    .enumerate()
                    // and calculate the overall bit index
                    .map(move |(bit_index, bit)| (bit_index + elem_index * 64, bit))
            })
            // only consider set bits
            .filter(|(_, bit)| *bit)
            // get their indices
            .map(|(index, _)| index)
    }
}

// Extracts and iterates over each bit in a u64
fn iterate_bits(value: u64) -> impl Iterator<Item = bool> {
    (0..u64::BITS)
        .into_iter()
        .map(move |i| (value >> i) & 1 == 1)
}
