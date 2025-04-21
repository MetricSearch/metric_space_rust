use anyhow::Result;
use bits::{f32_data_to_cubic_bitrep, hamming_distance};
use bitvec_simd::BitVecSimd;
use metrics::euc;
use ndarray::{Array1, ArrayView1, Axis};
use rayon::prelude::*;
use std::collections::HashSet;
use std::rc::Rc;
use std::time::Instant;
use wide::u64x4;
use dao::{Dao};
use dao::csv_dao_loader::{dao_from_csv_dir};
use utils::arg_sort_2d;
//use divan::Bencher;

fn main() -> Result<()> {
    tracing::info!("Loading mf dino data...");
    let num_queries = 10_000;
    let num_data = 1_000_000 - num_queries;

    let dao_f32: Rc<Dao<Array1<f32>>> = Rc::new(dao_from_csv_dir(
        "/Volumes/Data/RUST_META/mf_dino2_csv/",
        num_data,
        num_queries,
    )?);
    // just take 1 queries

    let queries = dao_f32.get_queries();
    let data = dao_f32.get_data();

    let (queries, _rest_queries) = queries.split_at(Axis(0),100);

    println!("Doing {:?} queries", queries.len());

    let data_bitreps = f32_data_to_cubic_bitrep(data);
    let queries_bitreps = f32_data_to_cubic_bitrep(queries);

    println!("Brute force NNs for {:?} queries", queries.len());
    let euc_dists: Vec<Vec<f32>> = brute_force_all_dists(queries, data);
    let (gt_nns, _gt_dists) = arg_sort_2d(euc_dists);  // these are all the sorted gt ids.

    // TEST code: just do one query for now with the data[0]
    // TEST code: let queries = dao.data.view().split_at( Axis(0), 1).0.to_owned();
    // TEST code: println!("queries size {:?}", queries.len());
    // TEST code: let nns_data_0 = brute_force_nns(&queries, &dao.data, 5);

    // println!("GT NNs for q0 = {:?} ", gt_nns.get(0).unwrap());

    println!("Running timings");
    let now = Instant::now();

    // Do a brute force of query bitmaps against the data bitmaps

    let hamming_distances = generate_hamming_dists(data_bitreps, queries_bitreps);
    let after = Instant::now();
    println!("Time per query 1_000_000 dists: {} ns", ((after - now).as_nanos() as f64) / num_queries as f64 );

    let (hamming_nns, _haming_dists) = arg_sort_2d(hamming_distances);

    for hamming_set_size in 10..51 { // was 100 to 550
        if hamming_set_size % 5 == 0 { // was 50
            report_queries(queries.len(), &gt_nns, &hamming_nns, hamming_set_size, hamming_set_size);
        }
    }

    Ok(())
}

fn report_queries(num_queries: usize, gt_nns: &Vec<Vec<usize>>, hamming_nns: &Vec<Vec<usize>>, hamming_set_size: usize, nns_size: usize) {
    println!("Benchmarking queries: hamming_set_size: {:?}", hamming_set_size);
    let mut sum = 0;
    let mut min = 100;
    let mut max = 0;
    (0..num_queries).into_iter().for_each(|qi| {

        let (hamming_nns, _rest_nns) = hamming_nns.get(qi).unwrap().split_at(hamming_set_size);
        let (gt_nns, _rest_gt_nns) = gt_nns.get(qi).unwrap().split_at(nns_size);

        // println!("Hamming NNs for q{} = {:?} ", query_index, hamming_500_nns );
        // println!("GT NNs for q{} = {:?} ", query_index, gt_100_nns );

        let hamming_set: HashSet<&usize> = HashSet::from_iter(hamming_nns);
        let gt_set: HashSet<&usize> = HashSet::from_iter(gt_nns);
        let intersection = hamming_set.intersection(&gt_set);

        let intersection_size = intersection.count();
        sum = sum + intersection_size;
        max = max.max(intersection_size);
        min = min.min(intersection_size);

        println!("Intersection of q{:?} {:?} Hamming vs {:?} nns, size: {:?}", qi, hamming_set_size, nns_size, intersection_size);
    }
    );
    println!("Mean size = {}, Max = {}, Min = {}", sum / num_queries , max, min);
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

fn generate_hamming_dists(
    data_bitreps: Vec<BitVecSimd<[u64x4; 4], 4>>,
    queries_bitreps: Vec<BitVecSimd<[u64x4; 4], 4>>,
) -> Vec<Vec<usize>> {
    queries_bitreps
        .par_iter()
        .map(|query| {
            data_bitreps
                .iter()
                .map(|data| hamming_distance(&query, &data))
                .collect::<Vec<usize>>()
        })
        .collect::<Vec<Vec<usize>>>()
}

