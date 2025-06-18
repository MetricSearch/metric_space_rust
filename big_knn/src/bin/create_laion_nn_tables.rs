/*
   First program to load data from Laion-400 h5 files.
   Load...
*/
use anyhow::Result;
use big_knn::{
    create_and_store_nn_table, get_file_names, get_file_sizes, make_partitions, partition_into,
};
use bits::container::Simd256x2;
use bits::EvpBits;
use clap::Parser;
use dao::hdf5_to_dao_loader::load_h5_files;
use dao::Dao;
use hdf5::File as Hdf5File;
use ndarray::{s, Array2, ArrayView1, ArrayView2};
use r_descent::IntoRDescent;
use std::fs;
use std::ops::Add;
use std::path::Path;
use std::rc::Rc;
use std::time::Instant;
use utils::arg_sort_big_to_small_2d;

#[global_allocator]
static GLOBAL: jemallocator::Jemalloc = jemallocator::Jemalloc;

/// clap parser
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Path to HDF5 source
    source_path: String,
    output_path: String,
}

const NUM_NEIGHBOURS: usize = 18;
const DELTA: f64 = 0.01;
const REVERSE_LIST_SIZE: usize = 32;
const NUM_VERTICES: usize = 333;
const UNUSED_DELETE_ME: usize = 0;

pub fn main() -> Result<()> {
    pretty_env_logger::formatted_timed_builder()
        .filter_level(log::LevelFilter::Trace)
        .init();

    let args = Args::parse();

    log::info!("Loading Laion-400M data...");
    let start = Instant::now();

    let dir_base = Path::new(&args.source_path);
    if !dir_base.is_dir() {
        anyhow::bail!("{} is not a directory", args.source_path);
    }

    let file_names = get_file_names(dir_base).unwrap();
    let sizes = get_file_sizes(dir_base, &file_names).unwrap();
    let partition_boundaries = partition_into(&sizes, 2_500_000).unwrap();
    let parititions = make_partitions(partition_boundaries, &file_names, &sizes);

    for i in 0..parititions.len() {
        println!("Partition: {}", i);
        let vec = &parititions[i];
        println!("Files in part: {:?}", vec);
    }

    for i in 0..parititions.len() {
        let part = parititions.get(i).unwrap();

        let dao = Rc::new(load_h5_files::<Simd256x2, 500>(dir_base, part, NUM_VERTICES).unwrap());

        let file_name = "nn_table".to_string().add(i.to_string().as_str());
        create_and_store_nn_table::<Simd256x2, 500>(
            dao,
            NUM_NEIGHBOURS,
            REVERSE_LIST_SIZE,
            UNUSED_DELETE_ME,
            DELTA,
            &args.output_path,
            &file_name,
        );
    }

    Ok(())
}
