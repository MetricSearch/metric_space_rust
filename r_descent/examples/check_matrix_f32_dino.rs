use anyhow::Result;
use bits::{f32_data_to_cubic_bitrep, whamming_distance};
use bitvec_simd::BitVecSimd;
use dao::csv_dao_matrix_loader::dao_matrix_from_csv_dir;
use dao::{Dao, DaoMatrix};
use metrics::euc;
use ndarray::Array1;
use r_descent_matrix::{get_nn_table2, initialise_table_m};
use std::collections::HashSet;
use std::fs::File;
use std::io::BufReader;
use std::rc::Rc;
use std::time::Instant;
use utils::dot_product_f32;

fn main() -> Result<()> {

    tracing::info!("Loading mf dino data...");
    let num_queries = 10_000;
    let num_data = 1_000_000 - num_queries;

    let data_file_name = "/Volumes/Data/RUST_META/mf_dino2_csv/";
    let descent_file_name = "_scratch/nn_table_100.bin";
    let rng_star_file_name = "_scratch/rng_table_100.bin";


    let start = Instant::now();

    println!("Loading mf dino data...");
    let num_queries = 10_000; // for runnning: 10_000;  // for testing 990_000
    let num_data = 1_000_000 - num_queries;
    let dao_f32: Rc<DaoMatrix<f32>> = Rc::new(dao_matrix_from_csv_dir(
        data_file_name,
        num_data,
        num_queries,
    )?);

    let start_post_load = Instant::now();

    let num_neighbours = 10;
    let chunk_size = 20000;
    let rho = 1.0;
    let delta = 0.01;
    let reverse_list_size = 32; // was 8

    println!("Initializing NN table with chunk size {}", chunk_size);
    let (mut ords,mut dists) = initialise_table_m(dao_f32.clone(), chunk_size, num_neighbours );

    for i in 0..3 {
        println!("Row {} ids: {:?} dists: {:?} ", i, ords.row(i), dists.row(i));
        for ord in ords.row(i) {
            println!( "ord: {} real dist is: {} ", ord, dot_product_f32(dao_f32.get_datum(i), dao_f32.get_datum(*ord)) );
        }
}

println!("Getting NN table");

    get_nn_table2(dao_f32.clone(), &mut ords, &mut dists, num_neighbours, rho, delta, reverse_list_size);

    println!("Line 0 of table:" );
    for i in 0..10 {
        println!(" neighbours: {} dists: {}", ords[[0,i]], dists[[0,i]] );
    }

    let end = Instant::now();

    println!("Finished (including load time in {} s", (end - start).as_secs());
    println!("Finished (post load time) in {} s", (end - start_post_load).as_secs());

    Ok(())
}





