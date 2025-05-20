use std::collections::HashSet;
use anyhow::Result;
use bits::Bsp;
use dao::Dao;
use ndarray::{s, Array1, Array2, ArrayView1};
use r_descent_matrix::{get_nn_table2_bsp, initialise_table_bsp};
use std::rc::Rc;
use std::time::Instant;
use dao::csv_dao_loader::dao_from_csv_dir;
use dao::pubmed_hdf5_gt_loader::hdf5_pubmed_gt_load;
use dao::pubmed_hdf5_to_dao_loader::hdf5_pubmed_f32_to_bsp_load;

fn main() -> Result<()> {
    println!("Loading Pubmed data...");

    let start = Instant::now();

    let f_name = "/Volumes/Data/sisap_challenge_25/pubmed/benchmark-dev-pubmed23.h5";

    let num_queries = 10_000;
    const ALL_RECORDS: usize = 0;
    const NUM_VERTICES: usize = 200;

    let dao_bsp: Rc<Dao<Bsp<2>>> = Rc::new(hdf5_pubmed_f32_to_bsp_load( f_name, ALL_RECORDS , num_queries, NUM_VERTICES ).unwrap());

    let queries: ArrayView1<Bsp<2>> = dao_bsp.get_queries();
    let data: ArrayView1<Bsp<2>> = dao_bsp.get_data();

    println!( "Pubmed data size: {} queries size: {}", data.len(), queries.len() );

    let start_post_load = Instant::now();

    let num_neighbours = 100; // was 10
    let chunk_size = 200;
    let rho = 1.0;
    let delta = 0.01;
    let reverse_list_size = 32;

    println!("Initializing NN table with chunk size {}", chunk_size);
    let (mut bsp_nns,mut bsp_dists) = initialise_table_bsp(dao_bsp.clone(), chunk_size, num_neighbours );

    println!("Getting NN table");

    get_nn_table2_bsp(dao_bsp.clone(), &mut bsp_nns, &mut bsp_dists, num_neighbours, rho, delta, reverse_list_size);

    println!("Line 0 of table:" );
    for i in 0..10 {
        println!(" neighbours: {} dists: {}", bsp_nns[[0,i]], bsp_dists[[0,i]] );
    }

    let end = Instant::now();

    println!("Finished (including load time in {} s", (end - start).as_secs());
    println!("Finished (post load time) in {} s", (end - start_post_load).as_secs());

    let knns = 100;

    let (gt_nns, gt_dists) = hdf5_pubmed_gt_load( f_name,knns ).unwrap();

    let dao_f32: Rc<Dao<Array1<f32>>> = Rc::new(dao_from_csv_dir(
        f_name,
        0,
        num_queries,
    )?);

    let gt_queries = dao_f32.get_queries();

    println!("Pubmed:");
    println!("results_size,gt_size,Mean,Max,Min,Std_dev" );
    for bsp_set_size in (30..101).step_by(5) {
        report_queries(gt_queries.len(), &gt_nns, &bsp_nns, bsp_set_size, 30);
    }

    Ok(())
}

fn report_queries(num_queries: usize, gt_nns: &Array2<usize>, bsp_nns: &Array2<usize>, bsp_set_size: usize, gt_size: usize) {

    let mut sum = 0;
    let mut min = usize::MAX;
    let mut max = 0;
    let mut intersection_sizes = Vec::with_capacity(num_queries);

    for qi in 0..num_queries {
        let bsp_row = &bsp_nns.row(qi);
        let gt_row = gt_nns.row(qi);

        if bsp_row.len() < bsp_set_size || gt_row.len() < gt_size {
            panic!("Not enough neighbors for query {}", qi);
        }

        let bsp_set: HashSet<usize> = bsp_row.slice( s![..bsp_set_size] ).iter().copied().collect();
        let gt_set: HashSet<usize> = gt_row.slice(s![..gt_size]).iter().copied().collect();

        let intersection_size = bsp_set.intersection(&gt_set).count();

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






