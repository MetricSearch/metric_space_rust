use std::ops::BitXor;
use bitvec_simd::BitVecSimd;
use ndarray::{Array1, ArrayView1};
use wide::u64x4;
use utils::arg_sort;

// First the cubic mappings

/// returns a Vector of BitVecSimds in which the bit in the bitvector is set if the corresponding value in the embedding space is positive.
/// Thus an input of
/// to_bitrep( [ [ 0.4, -0.3, 0.2 ], [ -0.9, -0.2, -0.1 ]] ) will create [[1,0,1],[0,0,0]]
pub fn f32_data_to_cubic_bitrep(embeddings: ArrayView1<Array1<f32>>) -> Vec<BitVecSimd<[wide::u64x4; 4], 4>> {
    embeddings
        .iter()
        .map(|embedding| f32_embedding_to_cubic_bitrep(embedding))
        .collect::<Vec<BitVecSimd<[wide::u64x4; 4], 4>>>()
}

pub fn f32_embedding_to_cubic_bitrep(embedding: &Array1<f32>) -> BitVecSimd<[wide::u64x4; 4], 4> {
    BitVecSimd::from_bool_iterator(embedding.iter().map(|&x| x < 0.0))
}

// The simple cubeoct mappings

// returns a Vector of BitVecSimds corresponding to standard cubeoctohedral mapping
pub fn f32_data_to_cubeoct_bitrep(embeddings: ArrayView1<Array1<f32>>) -> Vec<BitVecSimd<[wide::u64x4; 4], 4>> {
    embeddings
        .iter()
        .map(|embedding| f32_embedding_to_cubeoct_bitrep(embedding))
        .collect::<Vec<BitVecSimd<[wide::u64x4; 4], 4>>>()
}


pub fn f32_embedding_to_cubeoct_bitrep(embedding: &Array1<f32> ) -> BitVecSimd<[wide::u64x4; 4], 4> {

    let mut bit_vec = vec![];

    let (indices, _dists) = arg_sort(embedding.to_vec().iter().map(|x| { x.abs() } ).collect() );
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

// The evp 2 bit embeddings

pub fn f32_data_to_evp<const D: usize>(embeddings: ArrayView1<Array1<f32>>, non_zeros: usize) -> Vec<BitVecSimd<[u64x4; D],4>> {
    embeddings
        .iter()
        .map(|embedding| f32_embedding_to_evp(embedding,non_zeros))
        .collect::<Vec<BitVecSimd<[u64x4; D],4>>>()
}


pub fn f32_embedding_to_evp<const D: usize>(embedding: &Array1<f32>, non_zeros: usize) ->  BitVecSimd<[u64x4; D],4> {

    let mut bit_vec = vec![];

    let embedding_len = embedding.len();

    let (indices, _dists) = arg_sort(embedding.to_vec().iter().map(|x| { x.abs() } ).collect() );
    let ( _smallest_indices, biggest_indices ) = indices.split_at(embedding_len - non_zeros);

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

// The evp 5 bit embeddings

pub fn f32_data_to_hamming5bit<const D: usize>(embeddings: ArrayView1<Array1<f32>>, non_zeros: usize) -> Vec<BitVecSimd<[u64x4; D],4>> {
    embeddings
        .iter()
        .map(|embedding| f32_embedding_to_hamming5bit::<D>(embedding,non_zeros))
        .collect::<Vec<BitVecSimd<[u64x4; D],4>>>()
}

pub fn f32_embedding_to_hamming5bit<const D: usize>(embedding: &Array1<f32>, active_bits: usize) ->  BitVecSimd<[u64x4; D],4> {

    let mut bit_vec = vec![];

    let embedding_len = embedding.len();

    let (indices, _dists) = arg_sort(embedding.to_vec().iter().map(|x| { x.abs() } ).collect() );
    let ( _smallest_indices, biggest_indices ) = indices.split_at(embedding_len - active_bits);

    (0..embedding.len()).for_each( |index| {
        if biggest_indices.contains(&index) {
            if embedding[index] > 0.0 {         // +1
                bit_vec.push(false);
                bit_vec.push(true);
                bit_vec.push(true);
                bit_vec.push(true);
                bit_vec.push(true);
            } else {                            // -1
                bit_vec.push(false);
                bit_vec.push(false);
                bit_vec.push(false);
                bit_vec.push(false);
                bit_vec.push(false);
            }
        } else {                            // 0
            bit_vec.push(true);
            bit_vec.push(false);
            bit_vec.push(false);
            bit_vec.push(false);
            bit_vec.push(false);
        }
    });

    BitVecSimd::from_bool_iterator(bit_vec.into_iter())
}

fn iter_2bit_chunks<'a, const D: usize>(
    bv: &'a BitVecSimd<[wide::u64x4; D], 4>,
) -> impl Iterator<Item = u8> + 'a {
    let bit_len = bv.len();

    (0..bit_len / 2).map( |i| {
        let low = bv.get(i * 2).unwrap_or(false) as u8;
        let high = bv.get(i * 2 + 1).unwrap_or(false) as u8;
        low | (high << 1)
    })
}

// Bit Scalar Product

pub fn f32_embedding_to_bsp<const D: usize>(embedding: &Array1<f32>, non_zeros: usize) -> bsp<D> {
    let mut ones = vec![];
    let mut negative_ones = vec![];
    let embedding_len = embedding.len();

    let (indices, _dists) = arg_sort(embedding.to_vec().iter().map(|x| { x.abs() } ).collect() );
    let ( _smallest_indices, biggest_indices ) = indices.split_at(embedding_len - non_zeros);

    (0..embedding.len()).for_each( |index| {
        if biggest_indices.contains(&index) {
            if embedding[index] > 0.0 {
                ones.push(true);
            } else {
                ones.push(false);
            }
            if embedding[index] < 0.0 {
                negative_ones.push(true);
            } else {
                negative_ones.push(false);
            }
        } } );

    bsp::<D>{ ones: BitVecSimd::from_bool_iterator(ones.into_iter()),
        negative_ones: BitVecSimd::from_bool_iterator(negative_ones.into_iter()) }
}

// dot: a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()

// Real hamming distance:

pub fn hamming_distance<const D: usize>(
    a: &BitVecSimd<[wide::u64x4; D], 4>,
    b: &BitVecSimd<[wide::u64x4; D], 4>,
) -> usize {
    a.xor_cloned(&b).count_ones()
}

// This is weird hamming (whamming) distance recoded for the 00/11/01 (2 bit) encoding
pub fn whamming_distance<const D: usize>(
    a: &BitVecSimd<[wide::u64x4; D], 4>,
    b: &BitVecSimd<[wide::u64x4; D], 4>,
) -> usize {
    iter_2bit_chunks(a).zip(iter_2bit_chunks(b)).map( |(x, y)| {
        let hamm = x.bitxor(y);
        if hamm == 3 { 4 as usize } else { hamm as usize }
    } ).sum()
}

// BSP similarity

#[derive(Debug, Clone)]
pub struct bsp<const X: usize> {
    ones          : BitVecSimd<[u64x4; X],4>,
    negative_ones : BitVecSimd<[u64x4; X],4>,
}

pub fn bsp_similarity<const X: usize>( a: &bsp<X>, b: &bsp<X>) -> usize {
                                       // a : (BitVecSimd<[u64x4; X],4>,BitVecSimd<[u64x4; X],4>),
                                        // b : (BitVecSimd<[u64x4; X],4>,BitVecSimd<[u64x4; X],4>) ) -> usize {

    let A = a.ones.and_cloned(&b.ones).count_ones() as usize;
    let B = a.negative_ones.and_cloned(&b.negative_ones).count_ones() as usize;
    let C = a.ones.and_cloned(&b.negative_ones).count_ones() as usize;
    let D = b.ones.and_cloned(&a.negative_ones).count_ones() as usize;

    A + B - (C + D )
}


