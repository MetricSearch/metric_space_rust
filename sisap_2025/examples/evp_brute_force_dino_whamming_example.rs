use anyhow::Result;
use bits::{f32_data_to_hamming5bit, whamming_distance};
use bitvec_simd::BitVecSimd;
use dao::csv_dao_loader::dao_from_csv_dir;
use dao::Dao;
use metrics::euc;
use ndarray::{Array1, ArrayView1, Axis};
use rayon::prelude::*;
use std::collections::HashSet;
use std::rc::Rc;
use std::time::Instant;
use utils::arg_sort_2d;
use utils::index::Index;
use wide::u64x4;

fn main() -> Result<()> {
    tracing::info!("Loading mf dino data...");
    let num_queries = 10_000;
    let num_data = 1_000_000 - num_queries;

    let dao_f32: Rc<Dao<Array1<f32>>> = Rc::new(dao_from_csv_dir(
        "/Volumes/Data/RUST_META/mf_dino2_csv/",
        num_data,
        num_queries,
    )?);

    let queries: ArrayView1<Array1<f32>> = dao_f32.get_queries();
    let data: ArrayView1<Array1<f32>> = dao_f32.get_data();

    let (queries, _rest_queries) = queries.split_at(Axis(0), 100);

    println!("Doing {:?} queries", queries.len());

    let num_queries = queries.len() as f64;

    // This is a 2 bit encoding => need whamming distance

    let data_bitreps = f32_data_to_hamming5bit::<8>(data, 200); // 200 bits selected
    let queries_bitreps = f32_data_to_hamming5bit::<8>(queries, 200);

    println!("Brute force NNs for {} queries", queries.len());
    let now = Instant::now();
    let euc_dists: Vec<Vec<f32>> = brute_force_all_dists(queries, data);
    let after = Instant::now();

    println!(
        "Time per EUC query 1_000_000 dists: {} ns",
        ((after - now).as_nanos() as f64) / num_queries
    );

    let (gt_nns, gt_dists) = arg_sort_2d(euc_dists); // these are all the sorted gt ids.

    let now = Instant::now();

    // Do a brute force of query bitmaps against the data bitmaps

    let hamming_distances = generate_hamming_dists::<8>(queries_bitreps, data_bitreps);
    let after = Instant::now();

    println!(
        "Time per hamming query 1_000_000 dists: {} ns",
        ((after - now).as_nanos() as f64) / num_queries
    );

    let (hamming_nns, hamming_dists) = arg_sort_2d(hamming_distances);

    let gt_nns_0 = &gt_nns[0];
    let gt_dists_0 = &gt_dists[0];

    let bsp_nns_0 = &hamming_nns[0];
    let bsp_dists_0 = &hamming_dists[0];

    println!(
        "gt_nns gt_dists: {:?} {:?}",
        &gt_nns_0[0..10],
        &gt_dists_0[0..10]
    );

    println!(
        "wham_nns bsp_dists: {:?} {:?}",
        &bsp_nns_0[0..10],
        &bsp_dists_0[0..10]
    );

    for gt_size in (10..101).step_by(5) {
        report_queries(queries.len(), &gt_nns, &hamming_nns, 10, gt_size);
    }

    Ok(())
}

fn report_queries(
    num_queries: usize,
    gt_nns: &Vec<Vec<Index>>,
    hamming_nns: &Vec<Vec<Index>>,
    hamming_set_size: usize,
    gt_size: usize,
) {
    println!(
        "Benchmarking queries: hamming_set_size: {} gt_size: {}",
        hamming_set_size, gt_size
    );
    let mut sum = 0;
    let mut min = 100;
    let mut max = 0;
    (0..num_queries).into_iter().for_each(|qi| {
        let (hamming_nns, _rest_nns) = hamming_nns.get(qi).unwrap().split_at(hamming_set_size);
        let (gt_nns, _rest_gt_nns) = gt_nns.get(qi).unwrap().split_at(gt_size);

        let hamming_set = hamming_nns.into_iter().collect::<HashSet<_>>();
        let gt_set = gt_nns.into_iter().collect::<HashSet<_>>();

        let intersection = hamming_set.intersection(&gt_set);

        let intersection_size = intersection.count();
        sum = sum + intersection_size;
        max = max.max(intersection_size);
        min = min.min(intersection_size);

        // println!("Intersection of q{} {} hamming sists in {} gt_nns, intersection size: {}", qi, hamming_set_size, nns_size, intersection_size);
    });
    println!(
        "Mean intersection size = {}, Max = {}, Min = {}",
        sum / num_queries,
        max,
        min
    );
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

fn generate_hamming_dists<const D: usize>(
    queries_bitreps: Vec<BitVecSimd<[u64x4; D], 4>>,
    data_bitreps: Vec<BitVecSimd<[u64x4; D], 4>>,
) -> Vec<Vec<usize>> {
    queries_bitreps
        .par_iter()
        .map(|query| {
            data_bitreps
                .iter()
                .map(|data| whamming_distance::<D>(&query, &data))
                .collect::<Vec<usize>>()
        })
        .collect::<Vec<Vec<usize>>>()
}
