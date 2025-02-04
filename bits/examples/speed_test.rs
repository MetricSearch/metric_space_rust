use std::collections::HashSet;
use dao::Dao32;
use metrics::euc;
use anyhow::Result;
use std::rc::Rc;
use std::time::Instant;
use bitvec_simd::BitVecSimd;
use wide::u64x4;
use dao::csv_f32_loader::csv_f32_load;
use rayon::prelude::*;
use ndarray::{Array, Array1, Array2, ArrayView, ArrayView2, Axis, Ix1, Ix2};
use utils::arg_sort_2D;
use bits::{embedding_to_bitrep,hamming_distance};

use divan::{Bencher};


fn main() -> Result<()> {

    tracing::info!("Loading mf dino data...");
    let num_queries = 10_000;
    let num_data = 1_000_000 - num_queries;

    let dao: Rc<Dao32> = Rc::new(Dao32::dao_from_csv_dir("/Volumes/Data/RUST_META", num_data, num_queries)?);

    // just take 1 queries

    let queries = dao.get_queries(); //queries.view().split_at( Axis(0), 1).0.to_owned(); // first query
    let data = dao.get_data();

    println!("Doing {:?} queries", queries.nrows());

    let data_bitreps = data_to_bitrep(data);
    let queries_bitreps = data_to_bitrep(queries);

    println!("Brute force NNs for {:?} queries", queries.nrows());
    let euc_dists: Vec<Vec<f32>> = brute_force_all_dists(queries, data);
    let (gt_nns,gt_dists) = arg_sort_2D(euc_dists);

    // TEST code: just do one query for now with the data[0]
    // TEST code: let queries = dao.data.view().split_at( Axis(0), 1).0.to_owned();
    // TEST code: println!("queries size {:?}", queries.len());
    // TEST code: let nns_data_0 = brute_force_nns(&queries, &dao.data, 5);

    println!( "GT NNs for q0 = {:?} ", gt_nns.get(0).unwrap() );

    // println!("Running timings");
    // let now = Instant::now();

    let hamming_distances = generate_hamming_dists(data_bitreps, queries_bitreps);
    // let after = Instant::now();
    // println!("Time per query: {} ms", ((after - now).as_millis() as f64) / num_queries as f64 );

    let (hamming_nns,haming_dists) = arg_sort_2D(hamming_distances);

    println!( "Hamming NNs for q0 = {:?} ", hamming_nns.get(0).unwrap() );

    let hamming_set : HashSet<usize> =  HashSet::from_iter(hamming_nns.get(0).unwrap().iter().cloned());
    let gt_set : HashSet<usize> =  HashSet::from_iter(gt_nns.get(0).unwrap().iter().cloned());

    println!( "Intersection of nns is: {:?}", hamming_set.intersection(&gt_set) );

    Ok(())
}


//Returns the nn(k) using Euc as metric for queries
fn brute_force_all_dists(queries: ArrayView2<f32>, data: ArrayView2<f32>) -> Vec<Vec<f32>> {
   queries
       .axis_iter(Axis(0))
       .map(|q| {
           data
               .axis_iter(Axis(0))
               .map( |d| euc(q, d) )
           .collect()
       } ).collect()
}

fn generate_hamming_dists(data_bitreps: Vec<BitVecSimd<[u64x4; 4], 4>>, queries_bitreps: Vec<BitVecSimd<[u64x4; 4], 4>>) -> Vec<Vec<usize>> {
    queries_bitreps.par_iter().map(
        |query| {
            data_bitreps.iter().map(
                |data| {
                    hamming_distance(&query, &data)
                }
            )
                .collect::<Vec<usize>>()
        })
        .collect::<Vec<Vec<usize>>>()
}

/// returns a Vector of BitVecSimds in which the bit in the bitvector is set if the corresponding value in the embedding space is positive.
/// Thus an input of
/// to_bitrep( [ [ 0.4, -0.3, 0.2 ], [ -0.9, -0.2, -0.1 ]] ) will create [[1,0,1],[0,0,0]]
fn data_to_bitrep(embeddings: ArrayView2<f32>) -> Vec<BitVecSimd<[wide::u64x4; 4], 4>> {
    embeddings
        .axis_iter(Axis(0))
        .map( |embedding| embedding_to_bitrep(embedding) )
        .collect::<Vec<BitVecSimd<[wide::u64x4; 4], 4>>>()
}




