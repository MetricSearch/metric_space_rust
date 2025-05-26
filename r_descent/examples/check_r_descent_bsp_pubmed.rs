use anyhow::Result;
use bits::EVP_bits;
use chrono::Utc;
use dao::csv_dao_loader::dao_from_csv_dir;
use dao::pubmed_hdf5_gt_loader::hdf5_pubmed_gt_load;
use dao::pubmed_hdf5_to_dao_loader::{
    hdf5_pubmed_f32_to_bsp_load, hdf5_pubmed_f32_to_bsp_load_sequential,
};
use dao::Dao;
use ndarray::{s, Array1, Array2, ArrayView1};
use r_descent_matrix::{get_nn_table2_bsp, initialise_table_bsp};
use std::collections::HashSet;
use std::rc::Rc;
use std::time::Instant;

fn main() -> Result<()> {
    let start = Instant::now();
    let utc = Utc::now();
    println!("Started at {:?}", utc);
    println!("Loading Pubmed data...");

    let f_name = "/Volumes/Data/sisap_challenge_25/pubmed/benchmark-dev-pubmed23.h5";

    let num_queries = 10_000;
    const ALL_RECORDS: usize = 0;
    const NUM_VERTICES: usize = 200;

    let dao_bsp: Rc<Dao<EVP_bits<2>>> = Rc::new(
        hdf5_pubmed_f32_to_bsp_load(f_name, ALL_RECORDS, num_queries, NUM_VERTICES).unwrap(),
    );

    let queries: ArrayView1<EVP_bits<2>> = dao_bsp.get_queries();
    let data: ArrayView1<EVP_bits<2>> = dao_bsp.get_data();

    println!(
        "Pubmed data size: {} queries size: {}",
        data.len(),
        queries.len()
    );

    let start_post_load = Instant::now();

    let num_neighbours = 10;
    let chunk_size = 200;
    let rho = 1.0;
    let delta = 0.01;
    let reverse_list_size = 32;

    println!("Initializing NN table with chunk size {}", chunk_size);
    let (mut bsp_nns, mut bsp_dists) =
        initialise_table_bsp(dao_bsp.clone(), chunk_size, num_neighbours);

    println!("Getting NN table");

    get_nn_table2_bsp(
        dao_bsp.clone(),
        &mut bsp_nns,
        &mut bsp_dists,
        num_neighbours,
        rho,
        delta,
        reverse_list_size,
    );

    println!("Line 0 of table:");
    for i in 0..10 {
        println!(
            " neighbours: {} dists: {}",
            bsp_nns[[0, i]],
            bsp_dists[[0, i]]
        );
    }

    let end = Instant::now();

    println!(
        "Finished (including load time in {} s",
        (end - start).as_secs()
    );
    println!(
        "Finished (post load time) in {} s",
        (end - start_post_load).as_secs()
    );

    let knns = 30;

    let (gt_nns, gt_dists) = hdf5_pubmed_gt_load(f_name, knns).unwrap();

    Ok(())
}
