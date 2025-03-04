use bitvec_simd::BitVecSimd;
use ndarray::{Array1, Array2};

/*
 Also look at:
    bitm
    Crate bva: (no mention of SIMD)
        Struct Bvf: A bit vector using a statically allocated (stack allocated) memory implementation operates over: u8, u16, u32, u64, u128, usize
        Bv512
        Trait BitVector

 */
pub fn f32_embedding_to_bitrep(embedding: &Array1<f32>) -> BitVecSimd<[wide::u64x4; 4], 4> {
    BitVecSimd::from_bool_iterator(embedding.iter().map(|&x| x < 0.0))
}

// pub fn f32_matrix_to_bitrep_matrix(matrix: &Array2<f32>) -> Array2<> {
//
// }

// dot: a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()

pub fn hamming_distance(
    a: &BitVecSimd<[wide::u64x4; 4], 4>,
    b: &BitVecSimd<[wide::u64x4; 4], 4>,
) -> usize {
    //assert_eq!(a.len(), b.len());
    a.xor_cloned(&b).count_ones()
}
