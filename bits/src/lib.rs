use bitvec_simd::BitVecSimd;
use ndarray::{ArrayView, Ix1};

pub fn embedding_to_bitrep(embedding: ArrayView<f32, Ix1>) -> BitVecSimd<[wide::u64x4; 4], 4> {
    BitVecSimd::from_bool_iterator(embedding.iter().map(|&x| x < 0.0))
}

pub fn hamming_distance(
    a: &BitVecSimd<[wide::u64x4; 4], 4>,
    b: &BitVecSimd<[wide::u64x4; 4], 4>,
) -> usize {
    //assert_eq!(a.len(), b.len());
    a.xor_cloned(&b).count_ones()
}
