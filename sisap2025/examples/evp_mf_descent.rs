use anyhow::Result;
use bitvec_simd::BitVecSimd;
use dao::convert_f32_to_evp::f32_dao_to_evp;
use dao::csv_dao_loader::dao_from_csv_dir;
use dao::Dao;
use descent::Descent;
use ndarray::{Array1, Axis};
use std::rc::Rc;
use utils::pair::Pair;
use wide::u64x4;
use bits::evp::f32_data_to_evp;
//use std::time::Instant;

fn main() -> Result<()> {
    println!("Hello from MF EVP Descent");
    tracing_subscriber::fmt::init();
    //let now = Instant::now();
    tracing::info!("Loading mf dino data...");
    let num_queries = 10_000; // for runnning: 10_000;  // for testing 990_000
    let num_data = 1_000_000 - num_queries;
    let dao_f32: Rc<Dao<Array1<f32>>> = Rc::new(dao_from_csv_dir(
        "/Volumes/Data/RUST_META/mf_dino2_csv/",
        num_data,
        num_queries,
    )?);
    let num_neighbours = 10;
    //let max_candidates = 50;

    let dao_evp = f32_dao_to_evp::<3>(dao_f32.clone(), 100);

    let descent = Descent::new(dao_evp.clone(), num_neighbours, false, hamming_distance);

    println!("Built a NN table");

    let (queries, _rest_queries) = dao_f32.get_queries().split_at(Axis(0), 100); // get 100 queries

    println!("Doing {:?} queries", queries.len());

    let queries_bitreps = f32_data_to_evp::<3>(queries, 100);

    let mut results: Vec<Vec<Pair>> = vec![];

    for q in queries_bitreps {
        let (_canidates_len, sorted_results) = descent.knn_search::<BitVecSimd<[u64x4; 3], 4>>(
            q.clone(),
            &to_usize(&descent.current_graph.nns),
            dao_evp.clone(),
            100,
            hamming_distance,
        );
        results.push(sorted_results);
    }

    Ok(())
}

fn hamming_distance(a: &BitVecSimd<[u64x4; 3], 4>, b: &BitVecSimd<[u64x4; 3], 4>) -> f32 {
    a.xor_cloned(b).count_ones() as f32
}

pub fn to_usize(i32s: &Vec<Vec<i32>>) -> Vec<Vec<usize>> {
    i32s.into_iter()
        .map(|v| v.iter().map(|&v| v as usize).collect())
        .collect()
}
