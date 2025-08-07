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
}
