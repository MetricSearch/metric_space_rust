use std::collections::HashSet;
use anyhow::Result;
use bits::{f32_data_to_cubic_bitrep, hamming_distance};
use bitvec_simd::BitVecSimd;
use metrics::euc;
use ndarray::{Array1};
use std::fs::File;
use std::io::BufReader;
use std::rc::Rc;
use std::time::Instant;
use dao::{Dao};
use dao::csv_dao_loader::dao_from_csv_dir;
use r_descent::{getNNtable2, initialise_table};


fn main() -> Result<()> {
    tracing::info!("Loading mf dino data...");
    let num_queries = 10_000;
    let num_data = 1_000_000 - num_queries;

    let data_file_name = "/Volumes/Data/RUST_META/mf_dino2_csv/";
    let descent_file_name = "_scratch/nn_table_100.bin";
    let rng_star_file_name = "_scratch/rng_table_100.bin";

    println!("Loading mf dino data...");
    let num_queries = 10_000; // for runnning: 10_000;  // for testing 990_000
    let num_data = 1_000_000 - num_queries;
    let dao_f32: Rc<Dao<Array1<f32>>> = Rc::new(dao_from_csv_dir(
        data_file_name,
        num_data,
        num_queries,
    )?);

    let num_neighbours = 10;
    let chunk_size = 10_000;
    let rho = 1.0;
    let delta = 0.01;
    let reverse_list_size = 16;

    println!("Initializing NN table");
    let (mut ords,mut dists) = initialise_table( dao_f32.clone(),chunk_size,num_neighbours );

    println!( "After initialise_table" );
    println!("ords[1]: {:?}", ords[1]);
    println!("dists[1]: {:?}", dists[1]);

    // show(ords,dists);

    println!("Getting NN table");

    let (ords,dists) = getNNtable2(dao_f32.clone(), &mut ords, &mut dists, num_neighbours, rho, delta, reverse_list_size);

    println!("Line 1 of table:" );
    for i in 0..10 {
        println!(" neighbours: {} dists: {}", ords[0][i], dists[0][i] );
    }

    Ok(())
}

fn show(ords: Vec<Vec<usize>>, dists: Vec<Vec<f32>>) {
    println!("line 1: {:?} {:?} ", ords[1], dists[1]);
}




