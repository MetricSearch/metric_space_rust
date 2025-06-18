use bitvec_simd::BitVecSimd;
use ndarray::{Array1, ArrayView1};

/// returns a Vector of BitVecSimds in which the bit in the bitvector is set if the corresponding value in the embedding space is positive.
/// Thus an input of
/// to_bitrep( [ [ 0.4, -0.3, 0.2 ], [ -0.9, -0.2, -0.1 ]] ) will create [[1,0,1],[0,0,0]]
pub fn f32_data_to_cubic_bitrep(
    embeddings: ArrayView1<Array1<f32>>,
) -> Vec<BitVecSimd<[wide::u64x4; 4], 4>> {
    embeddings
        .iter()
        .map(f32_embedding_to_cubic_bitrep)
        .collect::<Vec<BitVecSimd<[wide::u64x4; 4], 4>>>()
}

pub fn f32_embedding_to_cubic_bitrep(embedding: &Array1<f32>) -> BitVecSimd<[wide::u64x4; 4], 4> {
    BitVecSimd::from_bool_iterator(embedding.iter().map(|&x| x < 0.0))
}
