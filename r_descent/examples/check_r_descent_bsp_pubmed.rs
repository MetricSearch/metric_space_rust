use anyhow::Result;
use bits::Bsp;
use dao::Dao;
use ndarray::{ArrayView1};
use r_descent_matrix::{get_nn_table2_bsp, initialise_table_bsp};
use std::rc::Rc;
use std::time::Instant;
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

    let num_neighbours = 10;
    let chunk_size = 200;
    let rho = 1.0;
    let delta = 0.01;
    let reverse_list_size = 32;

    println!("Initializing NN table with chunk size {}", chunk_size);
    let (mut ords,mut dists) = initialise_table_bsp(dao_bsp.clone(), chunk_size, num_neighbours );

    println!("Getting NN table");

    get_nn_table2_bsp(dao_bsp.clone(), &mut ords, &mut dists, num_neighbours, rho, delta, reverse_list_size);

    println!("Line 0 of table:" );
    for i in 0..10 {
        println!(" neighbours: {} dists: {}", ords[[0,i]], dists[[0,i]] );
    }

    let end = Instant::now();

    println!("Finished (including load time in {} s", (end - start).as_secs());
    println!("Finished (post load time) in {} s", (end - start_post_load).as_secs());

    Ok(())
}





