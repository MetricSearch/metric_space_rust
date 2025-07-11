use anyhow::Result;
use bits::container::{BitsContainer, Simd128};
use bits::{distance, similarity, EvpBits};
use dao::glove100_hdf5_dao_loader::hdf5_glove_f32_load;
use metrics::euc;
use ndarray::{Array1, ArrayView1, Axis};
use rayon::prelude::*;
use std::collections::HashSet;
use std::time::Instant;
use utils::arg_sort_2d;

fn main() -> Result<()> {
    tracing::info!("Loading Glove data...");

    let dao_f32 = hdf5_glove_f32_load("/Volumes/Data/glove-100-angular.hdf5").unwrap();

    let queries: ArrayView1<Array1<f32>> = dao_f32.get_queries();
    let data: ArrayView1<Array1<f32>> = dao_f32.get_data();

    println!(
        "Glove data size: {} queries size: {}",
        data.len(),
        queries.len()
    );

    let (queries, _rest_queries) = queries.split_at(Axis(0), 100);

    println!("Doing {:?} queries", queries.len());

    let num_queries = queries.len() as f64;

    // This is a 5 bit encoding => need hamming distance

    let data_bsp_reps = Array1::from_vec(
        data.into_par_iter()
            .map(|row| EvpBits::<Simd128, 100>::from_embedding(row.view(), 67))
            .collect::<Vec<_>>(),
    ); // 67 bits selected
    let queries_bsp_reps = Array1::from_vec(
        queries
            .into_par_iter()
            .map(|row| EvpBits::<Simd128, 100>::from_embedding(row.view(), 67))
            .collect::<Vec<_>>(),
    );

    println!("Brute force NNs for {} queries", queries.len());
    let now = Instant::now();
    let euc_dists: Vec<Vec<f32>> = brute_force_all_dists(queries, data);
    let after = Instant::now();

    println!(
        "Time per EUC query 1_000_000 dists: {} ns",
        ((after - now).as_nanos() as f64) / num_queries
    );

    let (gt_nns, _gt_dists) = arg_sort_2d(euc_dists); // these are all the sorted gt ids.

    let now = Instant::now();

    // Do a brute force of query bitmaps against the data bitmaps

    let hamming_distances = generate_bsp_dists(queries_bsp_reps, data_bsp_reps);
    let after = Instant::now();

    println!(
        "Time per BSP query 1_000_000 dists: {} ns",
        ((after - now).as_nanos() as f64) / num_queries
    );

    let (bsp_nns, _bsp_dists) = arg_sort_2d(hamming_distances);

    println!("Glove:");
    println!("results_size,gt_size,Mean,Max,Min,Std_dev");
    for bsp_set_size in (30..101).step_by(5) {
        report_queries(queries.len(), &gt_nns, &bsp_nns, bsp_set_size, 30);
    }

    Ok(())
}

fn report_queries(
    num_queries: usize,
    gt_nns: &Vec<Vec<usize>>,
    bsp_nns: &Vec<Vec<usize>>,
    bsp_set_size: usize,
    gt_size: usize,
) {
    let mut sum = 0;
    let mut min = 100;
    let mut max = 0;

    let mut intersection_sizes = vec![];
    (0..num_queries).into_iter().for_each(|qi| {
        let (hamming_nns, _rest_nns) = bsp_nns.get(qi).unwrap().split_at(bsp_set_size);
        let (gt_nns, _rest_gt_nns) = gt_nns.get(qi).unwrap().split_at(gt_size);

        let hamming_set: HashSet<usize> = hamming_nns.into_iter().map(|x| *x).collect();
        let gt_set: HashSet<usize> = gt_nns.into_iter().map(|x| *x).collect();

        let intersection = hamming_set.intersection(&gt_set);

        let intersection_size = intersection.count();
        sum = sum + intersection_size;
        max = max.max(intersection_size);
        min = min.min(intersection_size);
        intersection_sizes.push(intersection_size);

        // println!("Intersection of q{} {} hamming sists in {} gt_nns, intersection size: {}", qi, hamming_set_size, nns_size, intersection_size);
    });

    let mean = sum as f64 / num_queries as f64;
    println!(
        "{},{},{},{},{},{} ",
        bsp_set_size,
        gt_size,
        mean,
        max,
        min,
        std_dev(mean, intersection_sizes)
    );
}

fn std_dev(mean: f64, data: Vec<usize>) -> f64 {
    let variance = data
        .iter()
        .map(|value| {
            let diff = mean - *value as f64;
            diff * diff
        })
        .sum::<f64>()
        / data.len() as f64;

    variance.sqrt()
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
fn generate_bsp_dists<C: BitsContainer, const W: usize>(
    queries_bitreps: Array1<EvpBits<C, W>>,
    data_bitreps: Array1<EvpBits<C, W>>,
) -> Vec<Vec<usize>> {
    queries_bitreps
        .par_iter()
        .map(|query| {
            data_bitreps
                .iter()
                .map(|data| distance(&query, &data))
                .collect::<Vec<usize>>()
        })
        .collect::<Vec<Vec<usize>>>()
}
