
pub mod non_nan;
pub mod pair;

use std::rc::Rc;
use ndarray::{Array, Array1, Array2, ArrayView, ArrayView1, Axis, Ix1, ShapeBuilder};
use rand_distr::num_traits::Pow;
use crate::pair::Pair;
use crate::non_nan::NonNan;

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

pub fn min_index_and_value_v(vector: &Vec<f32>) -> (usize, f32) {
    let pair = vector
        .iter()
        .enumerate()
        .min_by(|best_so_far, to_compare| best_so_far.1.partial_cmp(to_compare.1)
            .unwrap())
        .unwrap();
    (pair.0,pair.1.clone())
}

pub fn index_of_min_v(vector: &Vec<f32>) -> usize {
    vector.iter().enumerate().min_by(|best_so_far, to_compare| best_so_far.1.partial_cmp(to_compare.1).unwrap()).unwrap().0
}

pub fn minimum_in_v(vector: &Vec<f32>) -> f32 {
    *vector.iter().min_by(|best_so_far, to_compare| best_so_far.partial_cmp(to_compare).unwrap()).unwrap()
}

pub fn min_index_and_value_a(arrai: &ArrayView1<f32>) -> (usize, f32) {
    let pair = arrai
        .iter()
        .enumerate()
        .min_by(|best_so_far, to_compare| best_so_far.1.partial_cmp(to_compare.1)
            .unwrap())
        .unwrap();
    (pair.0,pair.1.clone())
}

pub fn index_of_min_a(arrai: &ArrayView1<f32>) -> usize {
    arrai.iter().enumerate().min_by(|best_so_far, to_compare| best_so_far.1.partial_cmp(to_compare.1).unwrap()).unwrap().0
}

pub fn minimum_in_a(arrai: &ArrayView1<f32>) -> f32 {
    *arrai.iter().min_by(|best_so_far, to_compare| best_so_far.partial_cmp(to_compare).unwrap()).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    pub fn test_min_index_and_value() {
        let a = vec![-3.0, 0.0, -10.0, 5.0, -2.0];
        let (index,val) = min_index_and_value_v( &a );
        assert_eq!(val, -10.0);
        assert_eq!(index, 2);
    }

    #[test]
    pub fn test_index_of_min() {
        let a = vec![-3.0, 0.0, -10.0, 5.0, -2.0];
        assert_eq!(index_of_min_v( &a ), 2);
    }

    #[test]
    pub fn test_minimum_in() {
        let a = vec![-3.0, 0.0, -10.0, 5.0, -2.0];
        assert_eq!(minimum_in_v( &a ), -10.0);
    }
}

// Converts vectors of distances into vectors of indices and distances
// sorts into order from smaller to bigger.
pub fn arg_sort_small_to_big(dists: Array2<f32>) -> (Vec<Vec<usize>>, Vec<Vec<f32>>) {
    dists
        .axis_iter(Axis(0))
        .map(|row: ArrayView<f32,Ix1> | {
            let mut enumerated  = row.iter().enumerate().collect::<Vec<(usize, &f32)>>(); // Vec of positions (ords) and values (dists)
            enumerated.sort_by(|a, b| NonNan(*a.1).partial_cmp(&NonNan(*b.1)).unwrap());
            enumerated.into_iter().unzip()
        })
        .collect()
}

// Converts vectors of distances into vectors of indices and distances
// sorts into order from bigger to smaller
pub fn arg_sort_big_to_small(dists: Array2<f32>) -> (Vec<Vec<usize>>, Vec<Vec<f32>>) {
    dists
        .axis_iter(Axis(0))
        .map(|row: ArrayView<f32,Ix1> | {
            let mut enumerated  = row.iter().enumerate().collect::<Vec<(usize, &f32)>>(); // Vec of positions (ords) and values (dists)
            enumerated.sort_by(|a, b| NonNan(*b.1).partial_cmp(&NonNan(*a.1)).unwrap());
            enumerated.into_iter().unzip()
        })
        .collect()
}

// Converts 2d arrays of distances into 2d arrays of indices and distances
// sorts into order from smaller to bigger.
pub fn arg_sort_small_to_big_matrix(dists: Array2<f32>) -> (Array2<usize>, Array2<f32>) {
    let shape = dists.dim();

    let (ords, vals): (Vec<Vec<usize>>, Vec<Vec<f32>>) = dists
        .axis_iter(Axis(0))
        .map(|row: ArrayView<f32,Ix1> | {
            let mut enumerated  = row.iter().enumerate().collect::<Vec<(usize, &f32)>>(); // Vec of positions (ords) and values (dists)
            enumerated.sort_by(|a, b| NonNan(*a.1).partial_cmp(&NonNan(*b.1)).unwrap());
            enumerated.into_iter().unzip()
        })
        .unzip();

    let ords = Array2::from_shape_vec((shape.0, shape.1), ords.into_iter().flatten().collect()).unwrap();
    let vals = Array2::from_shape_vec((shape.0, shape.1), vals.into_iter().flatten().collect()).unwrap();

    (ords, vals)
}

// Converts 2d arrays of distances into 2d arrays of indices and distances
// sorts into order from bigger to smaller
pub fn arg_sort_big_to_small_matrix(dists: &Array2<f32>) -> (Array2<usize>, Array2<f32>) {
    let shape = dists.dim();

    let (ords, vals): (Vec<Vec<usize>>, Vec<Vec<f32>>) = dists
        .axis_iter(Axis(0))
        .map(|row: ArrayView<f32,Ix1> | {
            let mut enumerated  = row.iter().enumerate().collect::<Vec<(usize, &f32)>>(); // Vec of positions (ords) and values (dists)
            enumerated.sort_by(|a, b| NonNan(*b.1).partial_cmp(&NonNan(*a.1)).unwrap());
            enumerated.into_iter().unzip()
        })
        .unzip();

    let ords = Array2::from_shape_vec((shape.0, shape.1), ords.into_iter().flatten().collect()).unwrap();
    let vals = Array2::from_shape_vec((shape.0, shape.1), vals.into_iter().flatten().collect()).unwrap();

    (ords, vals)
}

// Converts vectors of distances into vectors of indices and distances
// sorts into order from smaller to bigger.
pub fn arg_sort<T: PartialOrd + Copy>(dists: Vec<T>) -> (Vec<usize>, Vec<T>) {
    let mut enumerated = dists.iter().enumerate().collect::<Vec<(usize, &T)>>();

    enumerated.sort_by(|a, b| a.1.partial_cmp(b.1).unwrap());

    enumerated.into_iter().unzip()
}

// Return the normalised DCG of two Vectors of results
pub fn ndcg(results : &Vec<Pair>, true_nns : &Vec<Pair> ) -> f32 {
    let num_true_nns = true_nns.len();
    let num_results = results.len();
    debug_assert!( num_true_nns == num_results );

    idcg(results, true_nns) / calc_norm_factor(true_nns.len())
}

fn calc_norm_factor(size: usize) -> f32 {
    let mut a_list = Vec::new();
    for i in 0..size { a_list.push(Pair::new(NonNan(i as f32), i * 100)); }
    idcg(&a_list,&a_list)
}
/* Ideal DCG */
fn idcg( results : &Vec<Pair>, true_nns : &Vec<Pair> ) -> f32 {
    let num_true_nns = true_nns.len();
    let num_results = results.len();
    debug_assert!( num_true_nns == num_results );

    let mut result = 0.0;
    for i in 0..num_results {
        let next_search_result = results.get(i).unwrap().index;
        match true_nns.iter().position( |x| x.index == next_search_result) {      // position of next result in true NNs
            Some(pos) => {
                let relevance = calc_relevance(pos as f32, num_true_nns as f32);
                result = result + f32::abs( relevance.pow(2.0) - 1.0) / ( f32::ln(i as f32) + 1.0 );},
            None =>  {}
        };

    }
    result
}

fn calc_relevance(correct_position : f32, num_nns : f32) -> f32 {
    let bottom = 1.0 + f32::exp(  - ( correct_position - (num_nns/2.0) ) );
    1.0 - ( 1.0 / bottom )
}


