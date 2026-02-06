use anyhow::Result;
use bitvec_simd::BitVecSimd;
//use rayon::prelude::*;
use dao::convert_f32_to_cube_oct::to_cube_oct_dao;
use dao::csv_dao_loader::dao_from_csv_dir;
use dao::Dao;
use descent::Descent;
use metrics::euc;
use ndarray::{Array1, ArrayView1};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::rc::Rc;
use std::time::Instant;
use utils::non_nan::NonNan;
use utils::pair::Pair;
use utils::{arg_sort_2d, ndcg};
use wide::u64x4;
//use divan::Bencher;

fn main() -> Result<()> {
    tracing::info!("Loading mf dino data...");

    let data_file_name = "/Volumes/Data/RUST_META/mf_dino2_csv/";
    let descent_file_name = "_scratch/nn_table_100.bin";
    // let rng_star_file_name = "_scratch/rng_table_100.bin";

    println!("cube-oct search:");
    println!("Serde load of Descent");
    let f = BufReader::new(File::open(descent_file_name).unwrap());
    let descent: Descent = bincode::deserialize_from(f).unwrap();

    // println!("Serde load code commented out for now");
    // let num_neighbours = 100;
    // let descent = Descent::new(dao.clone(), num_neighbours, true);

    println!("Loading mf dino data...");
    let num_queries = 10_000; // for runnning: 10_000;  // for testing 990_000
    let num_data = 1_000_000 - num_queries;

    let dao_f32: Rc<Dao<Array1<f32>>> =
        Rc::new(dao_from_csv_dir(data_file_name, num_data, num_queries)?);

    let dao_cube_oct = to_cube_oct_dao(dao_f32.clone());

    // let num_neighbours = 100;
    // let descent = Descent::new(dao_f32.clone(), num_neighbours, true);

    let queries = dao_cube_oct.get_queries().to_vec();
    let data = dao_cube_oct.get_data().to_vec();

    let this_many = 10;

    let (queries, _rest) = queries.split_at(this_many);

    let gt_pairs: Vec<Vec<Pair>> = brute_force_all_dists(queries.to_vec(), data, distance);

    let nn_table = to_usize(&descent.current_graph.nns);

    println!("Doing {:?} queries", queries.len());

    println!("Running queries");

    do_queries(
        queries.to_vec(),
        descent,
        dao_cube_oct.clone(),
        &gt_pairs,
        nn_table,
    );

    Ok(())
}

fn show_results(qid: usize, results: &Vec<Pair>) {
    print!("first few results for q{}:\t", qid);
    results.iter().by_ref().take(5).for_each(|pair| {
        print!("{} d: {} ", pair.index, pair.distance.as_f32());
    });
    println!();
}

fn show_gt(qid: usize, gt_pairs: &Vec<Vec<Pair>>) {
    //<<<<<<<<<<<<<<<<<
    print!("first few GT results for q{}:\t", qid);
    gt_pairs.get(qid).unwrap().iter().take(5).for_each(|pair| {
        print!("{} d: {} ", pair.index, pair.distance);
    });
    println!();
}

fn do_queries(
    queries: Vec<BitVecSimd<[u64x4; 4], 4>>,
    descent: Descent,
    dao: Rc<Dao<BitVecSimd<[u64x4; 4], 4>>>,
    gt_pairs: &Vec<Vec<Pair>>,
    nn_table: Vec<Vec<usize>>,
) {
    queries.iter().enumerate().for_each(|(qid, query)| {
        let now = Instant::now();
        let (dists, qresults) =
            descent.knn_search(query.clone(), &nn_table, dao.clone(), 100, distance);
        let after = Instant::now();
        println!("Results for Q{}....", qid);
        println!("Time per query: {} ns", (after - now).as_nanos());
        println!("Number of results = {} ", qresults.len());
        println!("Dists: {:?}", dists);
        show_results(qid, &qresults);
        show_gt(qid, gt_pairs);
        println!(
            "DCG: {}",
            ndcg(&qresults, &gt_pairs.get(qid).unwrap()[0..99])
        );
    });
}

// TODO fix this mess somehow!
fn to_usize(i32s: &Vec<Vec<i32>>) -> Vec<Vec<usize>> {
    i32s.into_iter()
        .map(|v| v.iter().map(|&v| v as usize).collect())
        .collect()
}

fn brute_force_all_dists<T: Clone>(
    queries: Vec<T>,
    data: Vec<T>,
    distance: fn(&T, &T) -> f32,
) -> Vec<Vec<Pair>> {
    queries
        .iter()
        .map(|q| {
            let mut pairs = data
                .iter()
                .enumerate()
                .map(|it| Pair::new(NonNan::new(distance(q, it.1)), it.0))
                .collect::<Vec<Pair>>();
            pairs.sort(); // Pair has Ord _by( |a, b| { a.distance.0.cmp(  b.distance.0 ) } );
            pairs
        })
        .collect::<Vec<Vec<Pair>>>()
}

fn distance(a: &BitVecSimd<[u64x4; 4], 4>, b: &BitVecSimd<[u64x4; 4], 4>) -> f32 {
    a.xor_cloned(b).count_ones() as f32
}
