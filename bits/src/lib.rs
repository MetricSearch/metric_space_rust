use bitvec_simd::BitVecSimd;
use deepsize::DeepSizeOf;
use ndarray::parallel::prelude::*;
use ndarray::{Array1, Array2, ArrayView1, Axis};
use rayon::iter::ParallelBridge;
use std::hash::Hasher;
use std::ops::BitXor;
use std::sync::Arc;
use utils::arg_sort;
use wide::u64x4;

// First the cubic mappings

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

// The simple cubeoct mappings

// returns a Vector of BitVecSimds corresponding to standard cubeoctohedral mapping
pub fn f32_data_to_cubeoct_bitrep(
    embeddings: ArrayView1<Array1<f32>>,
) -> Vec<BitVecSimd<[wide::u64x4; 4], 4>> {
    embeddings
        .iter()
        .map(f32_embedding_to_cubeoct_bitrep)
        .collect::<Vec<BitVecSimd<[wide::u64x4; 4], 4>>>()
}

pub fn f32_embedding_to_cubeoct_bitrep(embedding: &Array1<f32>) -> BitVecSimd<[wide::u64x4; 4], 4> {
    let mut bit_vec = vec![];

    let (indices, _dists) = arg_sort(embedding.to_vec().iter().map(|x| x.abs()).collect());
    let (_smallest_indices, biggest_indices) = indices.split_at(indices.len() / 2);

    (0..embedding.len()).for_each(|index| {
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

pub fn f32_data_to_evp<const D: usize>(
    embeddings: ArrayView1<Array1<f32>>,
    non_zeros: usize,
) -> Vec<BitVecSimd<[u64x4; D], 4>> {
    embeddings
        .iter()
        .map(|embedding| f32_embedding_to_evp(embedding, non_zeros))
        .collect::<Vec<BitVecSimd<[u64x4; D], 4>>>()
}

pub fn f32_embedding_to_evp<const D: usize>(
    embedding: &Array1<f32>,
    non_zeros: usize,
) -> BitVecSimd<[u64x4; D], 4> {
    let mut bit_vec = vec![];

    let embedding_len = embedding.len();

    let (indices, _dists) = arg_sort(embedding.to_vec().iter().map(|x| x.abs()).collect());
    let (_smallest_indices, biggest_indices) = indices.split_at(embedding_len - non_zeros);

    (0..embedding.len()).for_each(|index| {
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

pub fn f32_data_to_hamming5bit<const D: usize>(
    embeddings: ArrayView1<Array1<f32>>,
    non_zeros: usize,
) -> Vec<BitVecSimd<[u64x4; D], 4>> {
    embeddings
        .iter()
        .map(|embedding| f32_embedding_to_hamming5bit::<D>(embedding, non_zeros))
        .collect::<Vec<BitVecSimd<[u64x4; D], 4>>>()
}

pub fn f32_embedding_to_hamming5bit<const D: usize>(
    embedding: &Array1<f32>,
    active_bits: usize,
) -> BitVecSimd<[u64x4; D], 4> {
    let mut bit_vec = vec![];

    let embedding_len = embedding.len();

    let (indices, _dists) = arg_sort(embedding.to_vec().iter().map(|x| x.abs()).collect());
    let (_smallest_indices, biggest_indices) = indices.split_at(embedding_len - active_bits);

    (0..embedding.len()).for_each(|index| {
        if biggest_indices.contains(&index) {
            if embedding[index] > 0.0 {
                // +1
                bit_vec.push(false);
                bit_vec.push(true);
                bit_vec.push(true);
                bit_vec.push(true);
                bit_vec.push(true);
            } else {
                // -1
                bit_vec.push(false);
                bit_vec.push(false);
                bit_vec.push(false);
                bit_vec.push(false);
                bit_vec.push(false);
            }
        } else {
            // 0
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

    (0..bit_len / 2).map(|i| {
        let low = bv.get(i * 2).unwrap_or(false) as u8;
        let high = bv.get(i * 2 + 1).unwrap_or(false) as u8;
        low | (high << 1)
    })
}

// dot: a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()

// Real hamming distance:

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

// This is weird hamming (whamming) distance recoded for the 00/11/01 (2 bit) encoding
pub fn whamming_distance<const D: usize>(
    a: &BitVecSimd<[wide::u64x4; D], 4>,
    b: &BitVecSimd<[wide::u64x4; D], 4>,
) -> usize {
    iter_2bit_chunks(a)
        .zip(iter_2bit_chunks(b))
        .map(|(x, y)| {
            let hamm = x.bitxor(y);
            if hamm == 3 {
                4
            } else {
                hamm as usize
            }
        })
        .sum()
}

// Bit Scalar Product

#[derive(Debug, Clone)]
pub struct EvpBits<const X: usize> {
    pub ones: BitVecSimd<[u64x4; X], 4>,
    pub negative_ones: BitVecSimd<[u64x4; X], 4>,
    pub max_index: u8,
}

impl<const X: usize> Hasher for EvpBits<X> {
    fn finish(&self) -> u64 {
        self.ones
            .clone()
            .into_usizes()
            .iter()
            .map(|&x| x as u64)
            .sum::<u64>()
            + self
                .negative_ones
                .clone()
                .into_usizes()
                .iter()
                .map(|&x| x as u64)
                .sum::<u64>()
    }

    fn write(&mut self, _bytes: &[u8]) {}
}

impl<const X: usize> Default for EvpBits<X> {
    fn default() -> Self {
        Self {
            ones: BitVecSimd::from_slice(&[0]),
            negative_ones: BitVecSimd::from_slice(&[0]),
            max_index: 0
        }
    }
}

impl<const X: usize> DeepSizeOf for EvpBits<X> {
    fn deep_size_of_children(&self, _context: &mut deepsize::Context) -> usize {
        self.ones.storage_capacity() * size_of::<[u64x4; X]>()
            + size_of::<usize>()
            + self.negative_ones.storage_capacity() * size_of::<[u64x4; X]>()
            + size_of::<usize>()
    }
}

pub fn f32_embeddings_to_bsp<const D: usize>(
    embeddings: &Array2<f32>,
    non_zeros: usize,
) -> Array1<EvpBits<D>> {
    Array1::from_vec(
        embeddings
            .axis_iter(Axis(0))
            .into_par_iter()
            .map(|row| f32_embedding_to_bsp::<D>(&row, non_zeros))
            .collect::<Vec<_>>(),
    )
}

pub fn f32_embedding_to_bsp<const D: usize>(
    embedding: &ArrayView1<f32>,
    non_zeros: usize,
) -> EvpBits<D> {
    let mut ones = vec![];
    let mut negative_ones = vec![];
    let embedding_len = embedding.len();

    let (indices, _dists) = arg_sort(embedding.to_vec().iter().map(|x| x.abs()).collect());

    let (_smallest_indices, biggest_indices) = indices.split_at(embedding_len - non_zeros);

    (0..embedding.len()).for_each(|index| {
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
        } else {
            ones.push(false);
            negative_ones.push(false);
        }
    });

    EvpBits::<D> {
        ones: BitVecSimd::from_bool_iterator(ones.into_iter()),
        negative_ones: BitVecSimd::from_bool_iterator(negative_ones.into_iter()),
        max_index: indices[indices.len() - 1] as u8,
    }
}

pub fn f32_data_to_bsp<const D: usize>(
    embeddings: ArrayView1<Array1<f32>>,
    non_zeros: usize,
) -> Vec<EvpBits<D>> {
    embeddings
        .iter()
        .map(|embedding| f32_embedding_to_bsp::<D>(&embedding.view(), non_zeros))
        .collect::<Vec<EvpBits<D>>>()
}

#[inline(always)]
pub fn bsp_similarity<const X: usize>(a: &EvpBits<X>, b: &EvpBits<X>) -> usize {
    let aa = a.ones.and_cloned(&b.ones).count_ones();
    let bb = a.negative_ones.and_cloned(&b.negative_ones).count_ones();
    let cc = a.ones.and_cloned(&b.negative_ones).count_ones();
    let dd = b.ones.and_cloned(&a.negative_ones).count_ones();

    // println!( "a {:?} b {:?}", a, b) ;

    // adding X * 256 * 2 means the result must be positive since second term is maximally X * 256 * 2  ( BitVecSimd<[u64x4; X],4> )
    // min is zero (not in practice) max is 2048 (not in practice).
    // println!("A={} B={} C={} D={} result ={}", A, B, C, D, (A + B+X*256*2) - (C + D ));

    (aa + bb + X * 256 * 2) - (cc + dd)
}

#[inline(always)]
pub fn bsp_similarity_as_f32<const X: usize>(a: &EvpBits<X>, b: &EvpBits<X>) -> f32 {
    let aa = a.ones.and_cloned(&b.ones).count_ones() as usize;
    let bb = a.negative_ones.and_cloned(&b.negative_ones).count_ones() as usize;
    let cc = a.ones.and_cloned(&b.negative_ones).count_ones() as usize;
    let dd = b.ones.and_cloned(&a.negative_ones).count_ones() as usize;

    // println!( "a {:?} b {:?}", a, b) ;

    // adding X * 256 * 2 means the result must be positive since second term is maximally X * 256 * 2  ( BitVecSimd<[u64x4; X],4> )
    // min is zero (not in practice) max is 2048 (not in practice).
    // println!("A={} B={} C={} D={} result ={}", A, B, C, D, (A + B+X*256*2) - (C + D ));

    ((aa + bb + X * 256 * 2) - (cc + dd)) as f32
}

#[inline(always)]
pub fn bsp_distance<const X: usize>(a: &EvpBits<X>, b: &EvpBits<X>) -> usize {
    let aa = a.ones.and_cloned(&b.ones).count_ones();
    let bb = a.negative_ones.and_cloned(&b.negative_ones).count_ones();
    let cc = a.ones.and_cloned(&b.negative_ones).count_ones();
    let dd = b.ones.and_cloned(&a.negative_ones).count_ones();

    (cc + dd + X * 256 * 2) - (aa + bb)
}

#[inline(always)]
pub fn bsp_distance_as_f32<const X: usize>(a: &EvpBits<X>, b: &EvpBits<X>) -> f32 {
    let aa = a.ones.and_cloned(&b.ones).count_ones();
    let bb = a.negative_ones.and_cloned(&b.negative_ones).count_ones();
    let cc = a.ones.and_cloned(&b.negative_ones).count_ones();
    let dd = b.ones.and_cloned(&a.negative_ones).count_ones();

    ((cc + dd + X * 256 * 2) - (aa + bb)) as f32
}

pub fn f32_embedding_to_i8_embedding(embedding: &ArrayView1<f32>, non_zeros: usize) -> Array1<i8> {
    let mut i_8_s = vec![];
    let embedding_len = embedding.len();

    let (indices, _dists) = arg_sort(embedding.to_vec().iter().map(|x| x.abs()).collect());

    let (_smallest_indices, biggest_indices) = indices.split_at(embedding_len - non_zeros);

    (0..embedding.len()).for_each(|index| {
        if biggest_indices.contains(&index) {
            if embedding[index] > 0.0 {
                // +1
                i_8_s.push(1);
            } else {
                // -1
                i_8_s.push(-1);
            }
        } else {
            // 0
            i_8_s.push(0);
        }
    });

    ndarray::Array1::from(i_8_s)
}

pub fn i8_similarity(a: ArrayView1<i8>, b: ArrayView1<i8>) -> usize {
    a.iter().zip(b.iter()).map(|(x, y)| (x * y) as usize).sum()
}

// should return the distance from each entry in A (as rows) to each in b.
// Matrix multiply: C = A × B using mult.
pub fn matrix_dot_bsp_sequential<const X: usize>(
    a: &ArrayView1<EvpBits<X>>,
    b: &ArrayView1<EvpBits<X>>,
    dot: fn(a: &EvpBits<X>, b: &EvpBits<X>) -> f32,
) -> Array2<f32> {
    let a_len = a.len();
    let b_len = b.len();

    let mut result = unsafe { Array2::<f32>::uninit((a_len, b_len)).assume_init() };

    a.iter()
        .enumerate()
        .for_each(|(a_index, a_item): (usize, &EvpBits<X>)| {
            b.iter()
                .enumerate()
                .for_each(|(b_index, b_item): (usize, &EvpBits<X>)| {
                    let loc = result.get_mut([a_index, b_index]).unwrap();
                    *loc = dot(a_item, b_item);
                });
        });

    result
}

pub fn matrix_dot_bsp<const X: usize>(
    a: &ArrayView1<EvpBits<X>>,
    b: &ArrayView1<EvpBits<X>>,
    dot: fn(a: &EvpBits<X>, b: &EvpBits<X>) -> f32,
) -> Array2<f32> {
    let a_len = a.len();
    let b_len = b.len();
    let b = Arc::new(b.to_owned()); // Arc for shared parallel use

    let result: Array2<f32> = Array2::from_shape_fn((a_len, b_len), |(_i, _j)| 0.0);

    // Parallel over rows of `a`
    let mut result = result;
    result
        .outer_iter_mut()
        .enumerate()
        .par_bridge()
        .for_each(|(i, mut row)| {
            let a_item = &a[i];
            for (j, b_item) in b.iter().enumerate() {
                row[j] = dot(a_item, b_item);
            }
        });

    result
}
