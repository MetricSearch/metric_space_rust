
pub mod non_nan;
pub mod pair;

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

// Converts vectors of distances into vectors of indices and distances
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
