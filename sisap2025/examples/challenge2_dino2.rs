/*
In this task, participants are asked to develop memory-efficient indexing solutions that will be used to compute an approximation of the k-nearest neighbor graph for k=15. Each solution will be run in a Linux container with limited memory and storage resources.
Container specifications: 8 virtual CPUs, 16 GB of RAM, the dataset will be mounted read-only into the container.
Wall clock time for the entire task: 12 hours.
Minimum average recall to be considered in the final ranking: 0.8.
The goal is to compute the k-nearest neighbor graph (without self-references), i.e., find the k-nearest neighbors using all objects in the dataset as queries.
We will measure graph’s quality as the recall against a provided gold standard and the full computation time (i.e., including preprocessing, indexing, and search, and postprocessing)
We provide a development dataset; the evaluation phase will use an undisclosed dataset of similar size computed with the same neural model.
*/

use anyhow::Result;
use bits::container::{Simd256p128, Simd256x2};
use bits::EvpBits;
use clap::Parser;
use dao::convert_f32_to_bsp::f32_dao_to_bsp;
use dao::csv_dao_matrix_loader::dao_matrix_from_csv_dir;
use dao::DaoMatrix;
use ndarray::{s, ArrayView1};
use r_descent::IntoRDescent;
use std::rc::Rc;
use std::time::Instant;

#[global_allocator]
static GLOBAL: jemallocator::Jemalloc = jemallocator::Jemalloc;

const NUM_NEIGHBOURS: usize = 18;
const CHUNK_SIZE: usize = 200;
const DELTA: f64 = 0.01;
const REVERSE_LIST_SIZE: usize = 32;

const _ALL_RECORDS: usize = 0;
const _NUM_VERTICES: usize = 256;
const NUM_QUERIES: usize = 0;

/// clap parser
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Path to HDF5
    path: String,
}

fn main() -> Result<()> {
    pretty_env_logger::formatted_timed_builder()
        .filter_level(log::LevelFilter::Trace)
        .init();

    let args = Args::parse();

    log::info!("Loading DINO2 data...");
    log::debug!("{}", size_of::<EvpBits<Simd256x2, 384>>());
    let start = Instant::now();

    let dao_f32: Rc<DaoMatrix<f32>> =
        Rc::new(dao_matrix_from_csv_dir(&args.path, 1_000_000, NUM_QUERIES)?);

    let dao_bsp = f32_dao_to_bsp::<Simd256x2, 384>(dao_f32.clone(), 200);
    let data = dao_bsp.get_data();

    log::info!(
        "Dino2 data size: {} | num data: {}",
        data.len(),
        dao_bsp.num_data,
    );

    let start_post_load = Instant::now();

    log::info!("Getting NN table");

    let descent = dao_bsp
        .clone()
        .into_rdescent(NUM_NEIGHBOURS, REVERSE_LIST_SIZE, DELTA);

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
    println!("====== Printing First 1000 Rows ======");
    for i in 0..10 {
        println!(
            "{:?},",
            descent
                .neighbours
                .row(i)
                .slice(s![0..])
                .iter()
                .map(|x| x + 1)
                .collect::<Vec<usize>>()
        );
    }

    Ok(())
}
