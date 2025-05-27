pub mod non_nan;
pub mod pair;

use crate::non_nan::NonNan;
use crate::pair::Pair;
use byte_unit::{AdjustedByte, Byte};
use ndarray::{
    parallel::prelude::*, Array1, Array2, ArrayBase, ArrayView, ArrayView1, Axis, Ix1, ViewRepr,
};
use rand::seq::index::sample;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::num_traits::Pow;
use std::sync::{LazyLock, Mutex};

const SEED: u64 = 323 * 162;
static RNG: LazyLock<Mutex<ChaCha8Rng>> =
    LazyLock::new(|| Mutex::new(ChaCha8Rng::seed_from_u64(SEED))); // random number

// Converts vectors of distances into vectors of indices and distances
pub fn arg_sort_2d<T: PartialOrd + Copy>(dists: Vec<Vec<T>>) -> (Vec<Vec<usize>>, Vec<Vec<T>>) {
    dists
        .iter()
        .map(|vec| {
            let mut enumerated = vec.iter().enumerate().collect::<Vec<(usize, &T)>>();

            enumerated.sort_by(|a, b| a.1.partial_cmp(b.1).unwrap());

            enumerated.into_iter().unzip()
        })
        .collect()
}

pub fn min_index_and_value(arrai: &ArrayView1<f32>) -> (usize, f32) {
    let pair = arrai
        .iter()
        .enumerate()
        .min_by(|best_so_far, to_compare| best_so_far.1.partial_cmp(to_compare.1).unwrap())
        .unwrap();
    (pair.0, *pair.1)
}

pub fn index_of_min(arrai: &ArrayView1<f32>) -> usize {
    arrai
        .iter()
        .enumerate()
        .min_by(|best_so_far, to_compare| best_so_far.1.partial_cmp(to_compare.1).unwrap())
        .unwrap()
        .0
}

pub fn minimum_in(arrai: &ArrayView1<f32>) -> f32 {
    *arrai
        .iter()
        .min_by(|best_so_far, to_compare| best_so_far.partial_cmp(to_compare).unwrap())
        .unwrap()
}

// Vec versions of the functions above.
// pub fn min_index_and_value_v(vector: &Vec<f32>) -> (usize, f32) {
//     let pair = vector
//         .iter()
//         .enumerate()
//         .min_by(|best_so_far, to_compare| best_so_far.1.partial_cmp(to_compare.1)
//             .unwrap())
//         .unwrap();
//     (pair.0,pair.1.clone())
// }

// // Converts vectors of distances into vectors of indices and distances
// // sorts into order from smaller to bigger.
// pub fn arg_sort_small_to_big(dists: Array2<f32>) -> (Vec<Vec<usize>>, Vec<Vec<f32>>) {
//     dists
//         .axis_iter(Axis(0))
//         .map(|row: ArrayView<f32,Ix1> | {
//             let mut enumerated  = row.iter().enumerate().collect::<Vec<(usize, &f32)>>(); // Vec of positions (ords) and values (dists)
//             enumerated.sort_by(|a, b| NonNan(*a.1).partial_cmp(&NonNan(*b.1)).unwrap());
//             enumerated.into_iter().unzip()
//         })
//         .collect()
// }
//
// // Converts vectors of distances into vectors of indices and distances
// // sorts into order from bigger to smaller
// pub fn arg_sort_big_to_small(dists: Array2<f32>) -> (Vec<Vec<usize>>, Vec<Vec<f32>>) {
//     dists
//         .axis_iter(Axis(0))
//         .map(|row: ArrayView<f32,Ix1> | {
//             let mut enumerated  = row.iter().enumerate().collect::<Vec<(usize, &f32)>>(); // Vec of positions (ords) and values (dists)
//             enumerated.sort_by(|a, b| NonNan(*b.1).partial_cmp(&NonNan(*a.1)).unwrap());
//             enumerated.into_iter().unzip()
//         })
//         .collect()
// }
//
// pub fn index_of_min_v(vector: &Vec<f32>) -> usize {
//     vector.iter().enumerate().min_by(|best_so_far, to_compare| best_so_far.1.partial_cmp(to_compare.1).unwrap()).unwrap().0
// }
//
// pub fn minimum_in_v(vector: &Vec<f32>) -> f32 {
//     *vector.iter().min_by(|best_so_far, to_compare| best_so_far.partial_cmp(to_compare).unwrap()).unwrap()
// }
//
// #[cfg(test)]
// mod tests {
//     use super::*;
//
//     #[test]
//     pub fn test_min_index_and_value() {
//         let a = vec![-3.0, 0.0, -10.0, 5.0, -2.0];
//         let (index,val) = min_index_and_value_v( &a );
//         assert_eq!(val, -10.0);
//         assert_eq!(index, 2);
//     }
//
//     #[test]
//     pub fn test_index_of_min() {
//         let a = vec![-3.0, 0.0, -10.0, 5.0, -2.0];
//         assert_eq!(index_of_min_v( &a ), 2);
//     }
//
//     #[test]
//     pub fn test_minimum_in() {
//         let a = vec![-3.0, 0.0, -10.0, 5.0, -2.0];
//         assert_eq!(minimum_in_v( &a ), -10.0);
//     }
// }

// Converts 2d arrays of distances into 2d arrays of indices and distances
// sorts into order from smaller to bigger.
pub fn arg_sort_small_to_big(dists: Array2<f32>) -> (Array2<usize>, Array2<f32>) {
    let shape = dists.dim();

    let (ords, vals): (Vec<Vec<usize>>, Vec<Vec<f32>>) = dists
        .axis_iter(Axis(0))
        .map(|row: ArrayView<f32, Ix1>| {
            let mut enumerated = row.iter().enumerate().collect::<Vec<(usize, &f32)>>(); // Vec of positions (ords) and values (dists)
            enumerated.sort_by(|a, b| NonNan::new(*a.1).partial_cmp(&NonNan::new(*b.1)).unwrap());
            enumerated.into_iter().unzip()
        })
        .unzip();

    let ords =
        Array2::from_shape_vec((shape.0, shape.1), ords.into_iter().flatten().collect()).unwrap();
    let vals =
        Array2::from_shape_vec((shape.0, shape.1), vals.into_iter().flatten().collect()).unwrap();

    (ords, vals)
}

// Converts 2d arrays of distances into 2d arrays of indices and distances
// sorts into order from bigger to smaller
pub fn arg_sort_big_to_small(dists: &Array2<f32>) -> (Array2<usize>, Array2<f32>) {
    let shape = dists.dim();

    let (ords, vals): (Vec<Vec<usize>>, Vec<Vec<f32>>) = dists
        .axis_iter(Axis(0))
        .map(|row: ArrayView<f32, Ix1>| {
            arg_sort_big_to_small_1d(row)
        })
        .unzip();

    let ords =
        Array2::from_shape_vec((shape.0, shape.1), ords.into_iter().flatten().collect()).unwrap();
    let vals =
        Array2::from_shape_vec((shape.0, shape.1), vals.into_iter().flatten().collect()).unwrap();

    (ords, vals)
}

pub fn arg_sort_big_to_small_1d(dists: ArrayView<f32, Ix1>) -> (Vec<usize>, Vec<f32>) {
    let mut enumerated = dists.iter().enumerate().collect::<Vec<(usize, &f32)>>(); // Vec of positions (ords) and values (dists)
    enumerated.sort_by(|a, b| NonNan::new(*b.1).partial_cmp(&NonNan::new(*a.1)).unwrap());
    enumerated.into_iter().unzip()
}

// Converts vectors of distances into vectors of indices and distances
// sorts into order from smaller to bigger.
pub fn arg_sort<T: PartialOrd + Copy>(dists: Vec<T>) -> (Vec<usize>, Vec<T>) {
    let mut enumerated = dists.iter().enumerate().collect::<Vec<(usize, &T)>>();

    enumerated.sort_by(|a, b| a.1.partial_cmp(b.1).unwrap());

    enumerated.into_iter().unzip()
}

// Return the normalised DCG of two Vectors of results
pub fn ndcg(results: &[Pair], true_nns: &[Pair]) -> f32 {
    let num_true_nns = true_nns.len();
    let num_results = results.len();
    debug_assert!(num_true_nns == num_results);

    idcg(results, true_nns) / calc_norm_factor(true_nns.len())
}

fn calc_norm_factor(size: usize) -> f32 {
    let mut a_list = Vec::new();
    for i in 0..size {
        a_list.push(Pair::new(NonNan::new(i as f32), i * 100));
    }
    idcg(&a_list, &a_list)
}
/* Ideal DCG */
fn idcg(results: &[Pair], true_nns: &[Pair]) -> f32 {
    let num_true_nns = true_nns.len();
    let num_results = results.len();
    debug_assert!(num_true_nns == num_results);

    let mut result = 0.0;
    for i in 0..num_results {
        let next_search_result = results.get(i).unwrap().index;

        // position of next result in true NNs
        if let Some(pos) = true_nns.iter().position(|x| x.index == next_search_result) {
            let relevance = calc_relevance(pos as f32, num_true_nns as f32);
            result += f32::abs(relevance.pow(2.0) - 1.0) / (f32::ln(i as f32) + 1.0);
        }
    }
    result
}

fn calc_relevance(correct_position: f32, num_nns: f32) -> f32 {
    let bottom = 1.0 + f32::exp(-(correct_position - (num_nns / 2.0)));
    1.0 - (1.0 / bottom)
}

/*
    randperm(n,k) returns a vector containing k unique integers selected randomly from 1 to n.
*/
pub fn rand_perm(drawn_from: usize, how_many: usize) -> Array1<usize> {
    if drawn_from == 0 {
        return Array1::default([0]);
    }
    // Als old example:
    // let perm = RandomPermutation::with_rng(drawn_from as u64,&mut RNG.lock().unwrap() ).unwrap();
    // perm.iter().take(how_many).map(|x| x as usize).collect::<Array1<usize>>()

    let rng = &mut RNG.lock().unwrap();
    let sample = sample(rng, drawn_from, how_many).into_vec();
    Array1::from(sample)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    pub fn test_rnd_perm1() {
        let mut x = rand_perm(10, 10);
        x.to_vec().sort();
        assert_eq!(x.len(), 10);
        assert_eq!(x[0], 0);
        assert_eq!(x[5], 5);
        assert_eq!(x[9], 9);
    }
    #[test]
    pub fn test_rnd_perm2() {
        let mut y = rand_perm(10, 5);
        assert_eq!(y.len(), 5);
        assert!(y.iter().all(|&x| x >= 0 && x < 10));

        y.to_vec().sort();

        for i in 0..4 {
            assert!(y[i] < y[i + 1]);
        }
    }
}

// distances

pub fn distance_f32(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    f32::sqrt(a.iter().zip(b.iter()).map(|(a, b)| (a - b).powi(2)).sum())
}

pub fn dot_product_f32(
    a: ArrayBase<ViewRepr<&f32>, Ix1>,
    b: ArrayBase<ViewRepr<&f32>, Ix1>,
) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// Matrix multiply: C = A × B using mult.
pub fn matrix_dot_i8(
    a: &Array2<i8>,
    b: &Array2<i8>,
    mult: fn(a: &[i8], b: &[i8]) -> i32,
) -> Array2<i32> {
    let (m, k) = a.dim();
    let (bk, n) = b.dim();
    assert_eq!(k, bk);

    // Transpose B for column access
    let b_t = b.t();

    let mut result = Array2::<i32>::zeros((m, n));

    result
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .zip(a.axis_iter(Axis(0)))
        .for_each(|(mut row_c, row_a)| {
            for (j, col_b) in b_t.axis_iter(Axis(0)).enumerate() {
                row_c[j] = mult(row_a.as_slice().unwrap(), col_b.as_slice().unwrap());
            }
        });

    result
}

/// Number of bytes to human-readable `Display`able
pub fn bytes_fmt(num: usize) -> AdjustedByte {
    Byte::from(num).get_appropriate_unit(byte_unit::UnitType::Binary)
}
