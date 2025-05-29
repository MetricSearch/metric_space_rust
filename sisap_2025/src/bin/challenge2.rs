/*
In this task, participants are asked to develop memory-efficient indexing solutions that will be used to compute an approximation of the k-nearest neighbor graph for k=15. Each solution will be run in a Linux container with limited memory and storage resources.
Container specifications: 8 virtual CPUs, 16 GB of RAM, the dataset will be mounted read-only into the container.
Wall clock time for the entire task: 12 hours.
Minimum average recall to be considered in the final ranking: 0.8.
Dataset: GOOAQ (3 million vectors (384 dimensions) ).
The goal is to compute the k-nearest neighbor graph (without self-references), i.e., find the k-nearest neighbors using all objects in the dataset as queries.
We will measure graph’s quality as the recall against a provided gold standard and the full computation time (i.e., including preprocessing, indexing, and search, and postprocessing)
We provide a development dataset; the evaluation phase will use an undisclosed dataset of similar size computed with the same neural model.
*/

use std::time::Instant;

use anyhow::Result;
use bits::EvpBits;
use clap::Parser;
use dao::pubmed_hdf5_gt_loader::hdf5_pubmed_gt_load;
use dao::pubmed_hdf5_to_dao_loader::hdf5_pubmed_f32_to_bsp_load;
use dao::Dao;
use ndarray::{s, ArrayView1};
use r_descent::{get_nn_table2_bsp, initialise_table_bsp, IntoRDescent};
use std::rc::Rc;



/// clap parser
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Path to HDF5
    path: String,
}

fn main() -> Result<()>{
    pretty_env_logger::formatted_timed_builder()
        .filter_level(log::LevelFilter::Trace)
        .init();

    let args = Args::parse();

    log::info!("Loading GOOAQ data...");
    let start = Instant::now();

    const ALL_RECORDS: usize = 0;
    const NUM_VERTICES: usize = 256;
    const num_queries: usize = 0;

    let dao_bsp: Rc<Dao<EvpBits<2>>> = Rc::new(
        hdf5_pubmed_f32_to_bsp_load(&args.path, ALL_RECORDS, num_queries, NUM_VERTICES).unwrap(),
    );

    let data: ArrayView1<EvpBits<2>> = dao_bsp.get_data();

    log::info!(
        "GOOAQ data size: {} | num data: {}",
        data.len(),
        dao_bsp.num_data,
    );

    let start_post_load = Instant::now();

    let num_neighbours = 18;
    let chunk_size = 1000;
    let rho = 1.0;
    let delta = 0.01;
    let reverse_list_size = 64;

    log::info!("Getting NN table");

    let descent =
        dao_bsp
            .clone()
            .into_rdescent(num_neighbours, reverse_list_size, chunk_size, rho, delta);

    let end = Instant::now();

    log::info!(
        "Finished (including load time in {} s",
        (end - start).as_secs()
    );
    log::info!(
        "Finished (post load time) in {} s",
        (end - start_post_load).as_secs()
    );

    println!("***** Remember to add 1 to all results when returning for challenge!!");
    println!("====== Printing First 100 Rows ======");
    for i in 0..100 {
        println!("{:?}", descent.neighbours.row(i).slice(s![0..]).iter().map(|x| x + 1).collect::<Vec<usize>>());
    }

    Ok(())
}
