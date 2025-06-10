use std::hash::Hash;
use std::hash::Hasher;

pub use wide::{_256p128::_256p128, _256x2::_256x2, _256x4::_256x4, _256::_256};

//mod _512;
mod wide;

pub trait BitsContainer: Clone + Send + Sync {
    fn new() -> Self;
    fn count_ones(&self) -> usize;
    fn and_cloned(&self, other: &Self) -> Self;
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
