use anyhow::Result;
use bits::{bsp_similarity, i8_similarity, EvpBits};
use dao::pubmed_hdf5_gt_loader::hdf5_pubmed_gt_load;
use dao::pubmed_hdf5_to_i8_dao_loader::hdf5_pubmed_f32_to_i8_load;
use dao::{Dao, DaoMatrix};
use metrics::euc;
use ndarray::{s, Array2, ArrayView1, ArrayView2};
use rayon::iter::IntoParallelIterator;
use std::collections::HashSet;
use std::time::Instant;
use utils::arg_sort_2d;

fn main() -> Result<()> {
    let num_records = 0;
    let num_queries = 100;
    let vertices = 200;

    let f_name = "/Volumes/Data/sisap_challenge_25/pubmed/benchmark-dev-pubmed23.h5";

    tracing::info!("Loading Pubmed {} data...", num_records);

    let dao_bsp: DaoMatrix<i8> =
        hdf5_pubmed_f32_to_i8_load(f_name, num_records, num_queries, vertices).unwrap();

    let queries: ArrayView2<i8> = dao_bsp.get_queries();
    let data: ArrayView2<i8> = dao_bsp.get_data();

    println!(
        "Pubmed data size: {} queries size: {}",
        data.len(),
        queries.len()
    );

    let now = Instant::now();

    // Do a brute force of query bitmaps against the data bitmaps

    let i_8_distances = generate_i8_dists(queries, data);
    let after = Instant::now();

    println!(
        "Time per BSP query all dists: {} ns",
        ((after - now).as_nanos() as f64) / num_queries as f64
    );

    let (bsp_nns, _bsp_dists) = arg_sort_2d(i_8_distances);

    let bsp_nns = add_one(&bsp_nns);

    // println!( "First row of bsp dists: {:?}", &_bsp_dists[0][..20] );
    // println!( "First row of bsp ords: {:?}", &bsp_nns[0][..20] );

    let knns = 100;

    let (gt_nns, gt_dists) = hdf5_pubmed_gt_load(f_name, knns).unwrap();

    // println!( "First row of gt dists: {:?}", &gt_dists.row(0).slice(s![0..20]) );
    // println!( "First row of gt ords: {:?}", &gt_nns.row(0).slice(s![0..20]) );

    println!("Pubmed:");
    println!("results_size,gt_size,Mean,Max,Min,Std_dev");
    for bsp_set_size in (30..101).step_by(5) {
        report_queries(queries.len(), &gt_nns, &bsp_nns, bsp_set_size, 30);
    }

    Ok(())
}

fn add_one(ords: &Vec<Vec<usize>>) -> Vec<Vec<usize>> {
    ords.iter()
        .map(|row| row.iter().map(|entry| entry + 1).collect())
        .collect()
}

fn report_queries(
    num_queries: usize,
    gt_nns: &Array2<usize>,
    bsp_nns: &Vec<Vec<usize>>,
    bsp_set_size: usize,
    gt_size: usize,
) {
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

fn generate_i8_dists_explicit(queries: ArrayView2<i8>, datas: ArrayView2<i8>) -> Vec<Vec<usize>> {
    queries
        .rows()
        .into_iter()
        .map(|query| {
            datas
                .rows()
                .into_iter()
                .map(|data| (1 - i8_similarity(query, data)))
                .collect::<Vec<usize>>()
        })
        .collect::<Vec<Vec<usize>>>()
}

fn generate_i8_dists(queries: ArrayView2<i8>, datas: ArrayView2<i8>) -> Vec<Vec<usize>> {
    queries
        .dot(&datas.t())
        .rows()
        .into_iter()
        .map(|row| row.iter().map(|&x| x as usize).collect())
        .collect()
}
