use bitvec_simd::BitVecSimd;
use ndarray::{Array1, ArrayView1};
use utils::arg_sort;
/*
 Also look at:
    bitm
    Crate bva: (no mention of SIMD)
        Struct Bvf: A bit vector using a statically allocated (stack allocated) memory implementation operates over: u8, u16, u32, u64, u128, usize
        Bv512
        Trait BitVector

 */
pub fn f32_embedding_to_cubic_bitrep(embedding: &Array1<f32>) -> BitVecSimd<[wide::u64x4; 4], 4> {
    BitVecSimd::from_bool_iterator(embedding.iter().map(|&x| x < 0.0))
}

/// returns a Vector of BitVecSimds in which the bit in the bitvector is set if the corresponding value in the embedding space is positive.
/// Thus an input of
/// to_bitrep( [ [ 0.4, -0.3, 0.2 ], [ -0.9, -0.2, -0.1 ]] ) will create [[1,0,1],[0,0,0]]
pub fn f32_data_to_cubic_bitrep(embeddings: ArrayView1<Array1<f32>>) -> Vec<BitVecSimd<[wide::u64x4; 4], 4>> {
    embeddings
        .iter()
        .map(|embedding| f32_embedding_to_cubic_bitrep(embedding))
        .collect::<Vec<BitVecSimd<[wide::u64x4; 4], 4>>>()
}

pub fn f32_embedding_to_cubeoct_bitrep(embedding: &Array1<f32>) -> BitVecSimd<[wide::u64x4; 4], 4> {

    let mut bit_vec = vec![];

    let (indices, _dists) = arg_sort(embedding.to_vec());
    let ( _smallest_indices, biggest_indices ) = indices.split_at(indices.len() / 2);

    (0..embedding.len()).for_each( |index| {
        if biggest_indices.contains(&index) {
            if embedding[index] > 0.0 {
                bit_vec.push(true);
                bit_vec.push(true);
            } else {
                bit_vec.push(false);
                bit_vec.push(false);
            }
        } else {
            bit_vec.push(false);
            bit_vec.push(true);
        }
    });

    BitVecSimd::from_bool_iterator(bit_vec.into_iter())
}

// returns a Vector of BitVecSimds corresponding to standard cubeoctohedral mapping
pub fn f32_data_to_cubeoct_bitrep(embeddings: ArrayView1<Array1<f32>>) -> Vec<BitVecSimd<[wide::u64x4; 4], 4>> {
    embeddings
        .iter()
        .map(|embedding| f32_embedding_to_cubeoct_bitrep(embedding))
        .collect::<Vec<BitVecSimd<[wide::u64x4; 4], 4>>>()
}


// dot: a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()

pub fn hamming_distance(
    a: &BitVecSimd<[wide::u64x4; 4], 4>,
    b: &BitVecSimd<[wide::u64x4; 4], 4>,
) -> usize {
    //assert_eq!(a.len(), b.len());
    a.xor_cloned(&b).count_ones()
}
