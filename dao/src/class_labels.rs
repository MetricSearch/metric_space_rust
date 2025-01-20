use anndists::{dist::DistDot,prelude::*};
use rayon::prelude::*;

pub fn get_class_labels(
    hyperplane: &Vec<f32>,
    vectors: &Vec<Vec<f32>>,
    nn_table: &Vec<Vec<usize>>,
    alpha: u16,
) -> Vec<i32> {
    let dot_prod_over_data = vectors
        .par_iter()
        .map(|x| 1.0 - DistDot.eval(x.as_slice(), hyperplane.as_slice()))
        .collect::<Vec<f32>>();

    // Vector (indexed by ID) of 100NN vectors
    nn_table
        .par_iter()
        .map(|indexes| indexes.iter().filter_map(|&index| vectors.get(index))) // Maps NN indexes to vectors
        .map(|nn_vecs| nn_vecs.map(|x| 1.0 - DistDot.eval(x.as_slice(), hyperplane.as_slice()))) // Maps vectors to list of dot products
        .enumerate()
        .map(|(i, dots)| {
            dots.filter_map(|x| {
                if (x > 0.0) == (*dot_prod_over_data.get(i).unwrap() > 0.0) {
                    Some(1)
                } else {
                    None
                }
            }) // Counts the number on same side of HP as original item
                .sum::<u16>()
        })
        .map(|x| if x > alpha { 1 } else { 0 }) // Whether the sum is gt alpha
        .collect::<Vec<i32>>()
}
