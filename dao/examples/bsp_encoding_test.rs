use anyhow::Result;
use bits::{bsp, bsp_distance, bsp_similarity, f32_data_to_bsp, f32_data_to_cubeoct_bitrep, whamming_distance};
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
    // just take 1 query

    let queries = dao_f32.get_queries();
    let data = dao_f32.get_data();

    let (queries, _rest_queries) = queries.split_at(Axis(0),1);

    let data_bitreps = f32_data_to_bsp::<2>(data,200);
    let queries_bitreps = f32_data_to_bsp::<2>(queries,200);

    let data_0 =   data_bitreps.get(0).unwrap();
    let data_585585 =   data_bitreps.get(585585).unwrap();
    let data_1 =  data_bitreps.get(1).unwrap();
    let query_0 = queries_bitreps.get(0).unwrap();
    let data_2 =   data_bitreps.get(2).unwrap();

    println!("Data_0 has {} bits | XORed = {} bits", data_0.ones.count_ones() + data_0.negative_ones.count_ones(), data_0.ones.xor_cloned(&data_0.negative_ones).count_ones());
    println!("Data_1 has {} bits | XORed = {} bits", data_1.ones.count_ones() + data_1.negative_ones.count_ones(), data_1.ones.xor_cloned(&data_1.negative_ones).count_ones());

    println!("Object 0 bitrep ones: {:?} | negative ones: {:?}", data_0.ones, data_0.negative_ones);

    println!( "Smoking 0-0 distance {} similarity: {} ", bsp_distance::<2>(&data_0, &data_0), bsp_similarity::<2>(&data_0, &data_0) ); // two girls  --> 1024 + 200
    println!( "data 1-1 leaves distance {} similarity: {} ", bsp_distance::<2>(data_1, &data_1), bsp_similarity::<2>(data_1, &data_1) ); // smoking girl and leaves.
    println!( "data 2-2 leaves distance {} similarity: {} ", bsp_distance::<2>(data_2, &data_2), bsp_similarity::<2>(data_2, &data_2) ); // smoking girl and leaves.
    println!( "Smoking0-585 distance {} similarity: {} ", bsp_distance::<2>(data_585585, &data_0), bsp_similarity::<2>(data_585585, &data_0) ); // two girls
    println!( "Smoking 0-1 leaves distance {} similarity: {} ", bsp_distance::<2>(data_1, &data_0), bsp_similarity::<2>(data_1, &data_0) ); // smoking girl and leaves.
    println!( "Query 0-1 data distance {} similarity: {}  ", bsp_distance::<2>(query_0, &data_1), bsp_similarity::<2>(query_0, &data_1) ); // badness from other

    Ok(())
}

fn report_queries(num_queries: usize, gt_nns: &Vec<Vec<usize>>, hamming_nns: &Vec<Vec<usize>>, hamming_set_size: usize, gt_size: usize) {
    println!("Benchmarking queries: hamming_set_size: {} gt_size: {}", hamming_set_size, gt_size);
    let mut sum = 0;
    let mut min = 100;
    let mut max = 0;
    (0..num_queries).into_iter().for_each(|qi| {

        let (hamming_nns, _rest_nns) = hamming_nns.get(qi).unwrap().split_at(hamming_set_size);
        let (gt_nns, _rest_gt_nns) = gt_nns.get(qi).unwrap().split_at(gt_size);

        let hamming_set: HashSet<usize> = hamming_nns.into_iter().map(|x| *x ).collect();
        let gt_set: HashSet<usize> = gt_nns.into_iter().map(|x| *x ).collect();

        let intersection = hamming_set.intersection(&gt_set);

        let intersection_size = intersection.count();
        sum = sum + intersection_size;
        max = max.max(intersection_size);
        min = min.min(intersection_size);

        // println!("Intersection of q{} {} hamming sists in {} gt_nns, intersection size: {}", qi, hamming_set_size, nns_size, intersection_size);
    }
    );
    println!("Mean intersection size = {}, Max = {}, Min = {}", sum / num_queries , max, min);
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


fn generate_bsp_dists<const D: usize>(
    queries_bitreps: Vec<bsp<D>>,
    data_bitreps: Vec<bsp<D>>,
) -> Vec<Vec<usize>> {
    queries_bitreps
        .par_iter()
        .map(|query| {
            data_bitreps
                .iter()
                .map(|data| bsp_distance::<D>(&query, &data) )
                .collect::<Vec<usize>>()
        })
        .collect::<Vec<Vec<usize>>>()
}

