use crate::container::BitsContainer;
use bitvec_simd::BitVecSimd;
use ndarray::parallel::prelude::*;
use ndarray::{Array1, Array2, ArrayView1, Axis};
use rayon::iter::ParallelBridge;
use std::ops::BitXor;
use std::sync::Arc;
use utils::arg_sort;
use wide::u64x4;

pub use evp::{bsp_distance, bsp_similarity, EvpBits};

pub mod container;
pub mod evp;
pub mod cubic;
pub mod cubeoct;

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

pub fn f32_embeddings_to_bsp<C: BitsContainer, const W: usize>(
    embeddings: &Array2<f32>,
    non_zeros: usize,
) -> Array1<EvpBits<C, W>> {
    Array1::from_vec(
        embeddings
            .axis_iter(Axis(0))
            .into_par_iter()
            .map(|row| f32_embedding_to_bsp(&row, non_zeros))
            .collect::<Vec<_>>(),
    )
}

pub fn f32_embedding_to_bsp<C: BitsContainer, const W: usize>(
    embedding: &ArrayView1<f32>,
    non_zeros: usize,
) -> EvpBits<C, W> {
    let mut ones = C::new();
    let mut negative_ones = C::new();
    let embedding_len = embedding.len();

    let (indices, _dists) = arg_sort(embedding.to_vec().iter().map(|x| x.abs()).collect());

    let (_smallest_indices, biggest_indices) = indices.split_at(embedding_len - non_zeros);

    let mut one_index = 0;
    let mut negative_ones_index = 0;

    (0..embedding.len()).for_each(|index| {
        if biggest_indices.contains(&index) {
            if embedding[index] > 0.0 {
                ones.set_bit(one_index, true);
                one_index += 1;
            } else {
                ones.set_bit(one_index, false);
                one_index += 1;
            }
            if embedding[index] < 0.0 {
                negative_ones.set_bit(negative_ones_index, true);
                negative_ones_index += 1;
            } else {
                negative_ones.set_bit(negative_ones_index, false);
                negative_ones_index += 1;
            }
        } else {
            ones.set_bit(one_index, false);
            one_index += 1;

            negative_ones.set_bit(negative_ones_index, false);
            negative_ones_index += 1;
        }
    });

    EvpBits::<C, W>::new(ones, negative_ones)
}

pub fn f32_data_to_bsp<C: BitsContainer, const W: usize>(
    embeddings: ArrayView1<Array1<f32>>,
    non_zeros: usize,
) -> Vec<EvpBits<C, W>> {
    embeddings
        .iter()
        .map(|embedding| f32_embedding_to_bsp(&embedding.view(), non_zeros))
        .collect::<Vec<_>>()
}

#[inline(always)]
pub fn bsp_similarity_as_f32<C: BitsContainer, const W: usize>(
    a: &EvpBits<C, W>,
    b: &EvpBits<C, W>,
) -> f32 {
    bsp_similarity(a, b) as f32
}

#[inline(always)]
pub fn bsp_distance_as_f32<C: BitsContainer, const W: usize>(
    a: &EvpBits<C, W>,
    b: &EvpBits<C, W>,
) -> f32 {
    bsp_distance(a, b) as f32
}

// should return the distance from each entry in A (as rows) to each in b.
// Matrix multiply: C = A × B using mult.
pub fn matrix_dot_bsp_sequential<C: BitsContainer, const W: usize>(
    a: &ArrayView1<EvpBits<C, W>>,
    b: &ArrayView1<EvpBits<C, W>>,
    dot: fn(a: &EvpBits<C, W>, b: &EvpBits<C, W>) -> f32,
) -> Array2<f32> {
    let a_len = a.len();
    let b_len = b.len();

    let mut result = unsafe { Array2::<f32>::uninit((a_len, b_len)).assume_init() };

    a.iter().enumerate().for_each(|(a_index, a_item)| {
        b.iter().enumerate().for_each(|(b_index, b_item)| {
            let loc = result.get_mut([a_index, b_index]).unwrap();
            *loc = dot(a_item, b_item);
        });
    });

    result
}

pub fn matrix_dot_bsp<C: BitsContainer, const W: usize>(
    a: &ArrayView1<EvpBits<C, W>>,
    b: &ArrayView1<EvpBits<C, W>>,
    dot: fn(a: &EvpBits<C, W>, b: &EvpBits<C, W>) -> f32,
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
