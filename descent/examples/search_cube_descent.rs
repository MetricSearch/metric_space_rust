use anyhow::Result;
use bits::{f32_data_to_cubic_bitrep, whamming_distance};
use bitvec_simd::BitVecSimd;
use metrics::euc;
use ndarray::{Array1, ArrayView1};
//use rayon::prelude::*;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::rc::Rc;
use std::time::Instant;
use serde::__private::de::borrow_cow_bytes;
use wide::u64x4;
use dao::{Dao, DataType};
use dao::convert_f32_to_cubic::to_cubic_dao;
use dao::csv_dao_loader::dao_from_csv_dir;
use utils::{arg_sort_2d, ndcg};
use utils::non_nan::NonNan;
use descent::{Descent};
use utils::pair::Pair;
//use divan::Bencher;

fn main() -> Result<()> {
    tracing::info!("Loading mf dino data...");
    let num_queries = 10_000;
    let num_data = 1_000_000 - num_queries;

    let data_file_name = "/Volumes/Data/RUST_META/mf_dino2_csv/";
    let descent_file_name = "_scratch/nn_table_100.bin";
    let rng_star_file_name = "_scratch/rng_table_100.bin";

    println!("cubic search:");
    println!("Serde load of Descent");
    let f = BufReader::new(File::open(descent_file_name).unwrap());
    let descent: Descent = bincode::deserialize_from(f).unwrap();

    println!("Loading mf dino data...");
    let num_queries = 10_000; // for runnning: 10_000;  // for testing 990_000
    let num_data = 1_000_000 - num_queries;

    let dao_f32: Rc<Dao<Array1<f32>>> = Rc::new(dao_from_csv_dir(
        data_file_name,
        num_data,
        num_queries,
    )?);

    let dao_cube = to_cubic_dao(dao_f32.clone());

    let num_neighbours = 100;

    let queries = dao_cube.get_queries().to_vec();
    let data = dao_cube.get_data().to_vec();

    let this_many = 10;

    let (queries, _rest) = queries.split_at(this_many);

    let gt_pairs: Vec<Vec<Pair>> = brute_force_all_dists(queries.to_vec(), data);

    let nn_table = to_usize(&descent.current_graph.nns);

    println!("Doing {:?} queries", queries.len());

    println!("Running queries");

    do_queries(queries.to_vec(),descent,dao_cube.clone(),&gt_pairs,nn_table);

    Ok(())
}

fn show_results(qid : usize, results: &Vec<Pair>) {
    print!( "first few results for q{}:\t", qid );
    results
        .iter()
        .by_ref()
        .take(5)
        .for_each(|pair| { print!("{} d: {} ", pair.index, pair.distance.0 ); });
    println!();
}

fn show_gt(qid : usize, gt_pairs: &Vec<Vec<Pair>>) { //<<<<<<<<<<<<<<<<<
    print!( "first few GT results for q{}:\t", qid );
    gt_pairs
        .get(qid)
        .unwrap()
        .iter()
        .take(5)
        .for_each(|pair| {
            print!("{} d: {} ", pair.index, pair.distance );
        } );
    println!();

}

fn do_queries(
    queries: Vec<BitVecSimd<[u64x4; 4], 4>>,
    descent: Descent,
    dao: Rc<Dao<BitVecSimd<[u64x4; 4], 4>>>,
    gt_pairs: &Vec<Vec<Pair>>,
    nn_table: Vec<Vec<usize>>) {
    queries.
        iter().
        enumerate()
        .for_each( | (qid,query) | {
            let now = Instant::now();
            let (dists,qresults) = descent.knn_search( query.clone(), &nn_table, dao.clone(), 100 );
            let after = Instant::now();
            println!("Results for Q{}....", qid);
            println!("Time per query: {} ns", (after - now).as_nanos());
            println!("Number of results = {} ", qresults.len() );
            println!("Dists: {:?}", dists);
            show_results(qid,&qresults);
            show_gt(qid,gt_pairs);
            println!( "DCG: {}", ndcg(&qresults,
                                      &gt_pairs
                                          .get(qid)
                                          .unwrap()
                                          [0..99].into() ) );
        } );
}

// TODO fix this mess somehow!
fn to_usize(i32s: &Vec<Vec<i32>>) -> Vec<Vec<usize>> {
    i32s.into_iter().map(|v| v.iter().map(|&v| v as usize).collect()).collect()
}

fn brute_force_all_dists<T: Clone + DataType>(
    queries: Vec<T>,
    data: Vec<T>,
) -> Vec<Vec<Pair>> {
    queries
        .iter()
        .map( |q| {
            let mut pairs = data
                .iter()
                .enumerate()
                .map( |it| { Pair::new( NonNan(T::dist(q, it.1)), it.0 ) } )
                .collect::<Vec<Pair>>();
            pairs.sort(); // Pair has Ord _by( |a, b| { a.distance.0.cmp(  b.distance.0 ) } );
            pairs
        } )
        .collect::<Vec<Vec<Pair>>>()
}


