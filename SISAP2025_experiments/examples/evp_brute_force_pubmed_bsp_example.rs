use anyhow::Result;
use bits::{Bsp, bsp_similarity};
use metrics::euc;
use ndarray::{s, Array1, Array2, ArrayView1};
use rayon::prelude::*;
use std::collections::HashSet;
use std::time::Instant;
use dao::Dao;
use dao::pubmed_hdf5_dao_loader::hdf5_pubmed_f32_to_bsp_load;
use dao::pubmed_hdf5_gt_loader::hdf5_pubmed_gt_load;
use utils::arg_sort_2d;

fn main() -> Result<()> {

    let num_records = 0;
    let num_queries = 100;
    let vertices = 200;

    let f_name = "/Volumes/Data/sisap_challenge_25/pubmed/benchmark-dev-pubmed23.h5";

    tracing::info!("Loading Pubmed {} data...", num_records);

    let dao_bsp: Dao<Bsp<2>> = hdf5_pubmed_f32_to_bsp_load( f_name, num_records, num_queries, vertices ).unwrap();

    let queries: ArrayView1<Bsp<2>> = dao_bsp.get_queries();
    let data: ArrayView1<Bsp<2>> = dao_bsp.get_data();

    println!( "Pubmed data size: {} queries size: {}", data.len(), queries.len() );

    let now = Instant::now();

    // Do a brute force of query bitmaps against the data bitmaps

    let hamming_distances = generate_bsp_dists(queries, data);
    let after = Instant::now();

    println!("Time per BSP query 1_000_000 dists: {} ns", ((after - now).as_nanos() as f64) / num_queries as f64 );

    let (bsp_nns, _bsp_dists ) = arg_sort_2d(hamming_distances);

    let knns = 100;

    let gt_nns = hdf5_pubmed_gt_load( f_name,num_records,knns ).unwrap();

    println!("Pubmed:");
    println!("results_size,gt_size,Mean,Max,Min,Std_dev" );
    for bsp_set_size in (30..101).step_by(5) {
            report_queries(queries.len(), &gt_nns, &bsp_nns, bsp_set_size, 30);
    }

    Ok(())
}

fn report_queries(num_queries: usize, gt_nns: &Array2<usize>, bsp_nns: &Vec<Vec<usize>>, bsp_set_size: usize, gt_size: usize) {
    let mut sum = 0;
    let mut min = 100;
    let mut max = 0;

    let mut sum = 0;
    let mut min = usize::MAX;
    let mut max = 0;
    let mut intersection_sizes = Vec::with_capacity(num_queries);

    for qi in 0..num_queries {
        let bsp_row = &bsp_nns[qi];
        let gt_row = gt_nns.row(qi);

        if bsp_row.len() < bsp_set_size || gt_row.len() < gt_size {
            panic!("Not enough neighbors for query {}", qi);
        }

        let hamming_set: HashSet<usize> = bsp_row[..bsp_set_size].iter().copied().collect();
        let gt_set: HashSet<usize> = gt_row.slice(s![..gt_size]).iter().copied().collect();

        let intersection_size = hamming_set.intersection(&gt_set).count();

        sum += intersection_size;
        max = max.max(intersection_size);
        min = min.min(intersection_size);
        intersection_sizes.push(intersection_size);
    }
    
    let mean = sum as f64 / num_queries as f64;
    println!("{},{},{},{},{},{} ",  bsp_set_size, gt_size, mean, max, min, std_dev(mean,intersection_sizes) );
}

fn std_dev(mean: f64, data: Vec<usize>) -> f64 {
    let variance = data.iter()
        .map(|value| {
            let diff = mean - *value as f64;
            diff * diff
        })
        .sum::<f64>() / data.len() as f64;

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


fn generate_bsp_dists(
    queries_bitreps: ArrayView1<Bsp<2>>,
    data_bitreps: ArrayView1<Bsp<2>>,
) -> Vec<Vec<usize>> {
    queries_bitreps
        .iter()
        .map(|query| {
            data_bitreps
                .iter()
                .map(|data| ( 1 - bsp_similarity::<2>(query, data)) )
                .collect::<Vec<usize>>()
        })
        .collect::<Vec<Vec<usize>>>()
}


