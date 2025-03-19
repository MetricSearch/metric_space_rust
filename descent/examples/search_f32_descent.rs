use anyhow::Result;
use bits::{f32_data_to_cubic_bitrep, hamming_distance};
use bitvec_simd::BitVecSimd;
use metrics::euc;
use ndarray::{Array1, ArrayView1};
//use rayon::prelude::*;
use std::fs::File;
use std::io::BufReader;
use std::rc::Rc;
use std::time::Instant;
use wide::u64x4;
use dao::{Dao};
use dao::convert_f32_to_cubic::to_cubic_dao;
use dao::csv_f32_loader::dao_from_csv_dir;
use utils::arg_sort_2d;
use descent::non_nan::NonNan;
use descent::{Descent};
use descent::pair::Pair;
//use divan::Bencher;

fn main() -> Result<()> {
    tracing::info!("Loading mf dino data...");
    let num_queries = 10_000;
    let num_data = 1_000_000 - num_queries;

    let data_file_name = "/Volumes/Data/RUST_META/mf_dino2_csv/";
    let descent_file_name = "_scratch/nn_table_100.bin";
    let rng_star_file_name = "_scratch/rng_table_100.bin";

    println!("f32 search");
    println!("Serde load of Descent");
    let f = BufReader::new(File::open(descent_file_name).unwrap());
    let descent: Descent = bincode::deserialize_from(f).unwrap();

    // println!("Serde load code commented out for now");
    // let num_neighbours = 100;
    // let descent = Descent::new(dao.clone(), num_neighbours, true);

    println!("Loading mf dino data...");
    let num_queries = 10_000; // for runnning: 10_000;  // for testing 990_000
    let num_data = 1_000_000 - num_queries;
    let dao_f32: Rc<Dao<Array1<f32>>> = Rc::new(dao_from_csv_dir(
        data_file_name,
        num_data,
        num_queries,
    )?);

    let num_neighbours = 100;

    let queries = dao_f32.get_queries().to_vec();

    let this_many = 1000;

    let (queries, _rest) = queries.split_at(this_many);

    println!("Doing {:?} queries", queries.len());

    println!("Running timings");
    let now = Instant::now();

    let results = do_queries(queries,descent,dao_f32);

    let after = Instant::now();
    println!("Time per query: {} ms", ((after - now).as_millis() as f64) / this_many as f64 );

    show_results(results);

    Ok(())
}

fn show_results(results: Vec<Vec<Pair>>) {
    results.iter().enumerate().for_each(|(i, row)| {
        println!( "num results: {}", row.len());
    })
}

fn do_queries(
    queries_bitreps: &[Array1<f32>],
    descent: Descent,
    dao: Rc<Dao<Array1<f32>>>,
) -> Vec<Vec<Pair>> {
    let mut results: Vec<Vec<Pair>> = vec![];
    queries_bitreps
        .iter()
        .for_each(|query| {
            let (dists,qresults) = descent.knn_search( query.clone(), to_usize(&descent.current_graph.nns), dao.clone(), 100);
            println!("Dists: {:?}", dists);
            results.push(qresults);
        } );
    results
}

// TODO fix this mess somehow!
fn to_usize(i32s: &Vec<Vec<i32>>) -> Vec<Vec<usize>> {
    i32s.into_iter().map(|v| v.iter().map(|&v| v as usize).collect()).collect()
}

//Returns the nn(k) using Euc as metric for queries
fn brute_force_all_dists(
    queries: ArrayView1<Array1<f32>>,
    data: ArrayView1<Array1<f32>>,
) -> Vec<Vec<f32>> {
    queries
        .iter()
        .map(|q| data.iter().map(|d| euc(q, d)).collect())
        .collect()
}


