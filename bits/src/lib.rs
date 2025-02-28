use bitvec_simd::BitVecSimd;
use ndarray::{Array1};

pub fn f32_embedding_to_bitrep(embedding: &Array1<f32>) -> BitVecSimd<[wide::u64x4; 4], 4> {
    BitVecSimd::from_bool_iterator(embedding.iter().map(|&x| x < 0.0))
}

pub fn hamming_distance(
    a: &BitVecSimd<[wide::u64x4; 4], 4>,
    b: &BitVecSimd<[wide::u64x4; 4], 4>,
) -> usize {
    //assert_eq!(a.len(), b.len());
    a.xor_cloned(&b).count_ones()
}
