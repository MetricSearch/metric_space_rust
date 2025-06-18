use bitvec_simd::BitVecSimd;
pub use evp::{distance, similarity, EvpBits};

pub mod container;
pub mod cubeoct;
pub mod cubic;
pub mod evp;

pub fn hamming_distance<const D: usize>(
    a: &BitVecSimd<[wide::u64x4; D], 4>,
    b: &BitVecSimd<[wide::u64x4; D], 4>,
) -> usize {
    a.xor_cloned(b).count_ones()
}

pub fn hamming_distance_as_f32<const D: usize>(
    a: &BitVecSimd<[wide::u64x4; D], 4>,
    b: &BitVecSimd<[wide::u64x4; D], 4>,
) -> f32 {
    a.xor_cloned(b).count_ones() as f32
}
