use crate::container::BitsContainer;
use deepsize::DeepSizeOf;
use ndarray::{Array1, ArrayBase, ArrayView1, ArrayView2, Data, Ix1};
use ndarray::{Array2, Axis};
use rayon::iter::ParallelIterator;
use rayon::iter::{IntoParallelIterator, ParallelBridge};
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;
use utils::arg_sort;

/// Bit Scalar Product using bit container C, with actual width W
///
/// For example, an AVX512 container used to store 384 bits.
#[derive(Debug, Clone, Default)]
pub struct EvpBits<C, const W: usize> {
    ones: C,
    negative_ones: C,
}

impl<C: BitsContainer, const W: usize> EvpBits<C, W> {
    pub fn new(ones: C, negative_ones: C) -> Self {
        Self {
            ones,
            negative_ones,
        }
    }

    pub fn from_embedding<S: Data<Elem = f32>>(
        embedding: ArrayBase<S, Ix1>,
        non_zeros: usize,
    ) -> Self {
        assert_eq!(W, embedding.len());

        let mut ones = C::new();
        let mut negative_ones = C::new();
        let embedding_len = embedding.len();

        let (indices, _dists) = arg_sort(embedding.iter().map(|x| x.abs()).collect());

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

        Self::new(ones, negative_ones)
    }

    pub fn from_embeddings(embeddings: ArrayView2<f32>, non_zeros: usize) -> Array1<Self> {
        Array1::from_vec(
            embeddings
                .axis_iter(Axis(0))
                .into_par_iter()
                .map(|row| EvpBits::from_embedding(row, non_zeros))
                .collect::<Vec<_>>(),
        )
    }
}

impl<C, const W: usize> DeepSizeOf for EvpBits<C, W> {
    fn deep_size_of_children(&self, _context: &mut deepsize::Context) -> usize {
        size_of::<C>() * 2
    }
}

impl<C: BitsContainer, const W: usize> Hash for EvpBits<C, W> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.ones.hash::<_, W>(state);
        self.negative_ones.hash::<_, W>(state);
    }
}

#[inline(always)]
pub fn distance<C: BitsContainer, const W: usize>(a: &EvpBits<C, W>, b: &EvpBits<C, W>) -> usize {
    let aa = a.ones.and_cloned(&b.ones).count_ones();
    let bb = a.negative_ones.and_cloned(&b.negative_ones).count_ones();
    let cc = a.ones.and_cloned(&b.negative_ones).count_ones();
    let dd = b.ones.and_cloned(&a.negative_ones).count_ones();

    (cc + dd + W * 2) - (aa + bb)
}

#[inline(always)]
pub fn similarity<C: BitsContainer, const W: usize>(a: &EvpBits<C, W>, b: &EvpBits<C, W>) -> usize {
    let aa = a.ones.and_cloned(&b.ones).count_ones();
    let bb = a.negative_ones.and_cloned(&b.negative_ones).count_ones();
    let cc = a.ones.and_cloned(&b.negative_ones).count_ones();
    let dd = b.ones.and_cloned(&a.negative_ones).count_ones();

    // println!( "a {:?} b {:?}", a, b) ;

    // adding X * 256 * 2 means the result must be positive since second term is maximally X * 256 * 2  ( BitVecSimd<[u64x4; X],4> )
    // min is zero (not in practice) max is 2048 (not in practice).
    // println!("A={} B={} C={} D={} result ={}", A, B, C, D, (A + B+X*256*2) - (C + D ));

    (aa + bb + W * 2) - (cc + dd)
}

#[inline(always)]
pub fn similarity_as_f32<C: BitsContainer, const W: usize>(
    a: &EvpBits<C, W>,
    b: &EvpBits<C, W>,
) -> f32 {
    similarity(a, b) as f32
}

#[inline(always)]
pub fn distance_as_f32<C: BitsContainer, const W: usize>(
    a: &EvpBits<C, W>,
    b: &EvpBits<C, W>,
) -> f32 {
    distance(a, b) as f32
}

// Real hamming distance:
pub fn hamming_distance<C: BitsContainer>(a: &C, b: &C) -> usize {
    a.xor(b).count_ones()
}

pub fn hamming_distance_as_f32<C: BitsContainer>(a: &C, b: &C) -> f32 {
    hamming_distance(a, b) as f32
}

// should return the distance from each entry in A (as rows) to each in b.
// Matrix multiply: C = A × B using mult.
pub fn matrix_dot<C: BitsContainer, const W: usize>(
    a: ArrayView1<EvpBits<C, W>>,
    b: ArrayView1<EvpBits<C, W>>,
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
