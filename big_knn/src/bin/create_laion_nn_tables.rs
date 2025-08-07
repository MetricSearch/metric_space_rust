/*
   Program to create NN tables from Laion-400 h5 files.
*/
use anyhow::Result;
use big_knn::{
    create_and_store_nn_table, get_file_names, get_file_sizes, make_partitions, partition_into,
    DATA_DIM,
};
use bits::container::Simd256x2;
use bits::EvpBits;
use clap::Parser;
use dao::hdf5_to_dao_loader::load_h5_files;
use dao::Dao;
use hdf5::File as Hdf5File;
use ndarray::{s, Array2, ArrayView1, ArrayView2};
use r_descent::IntoRDescent;
use std::ops::Add;
use std::path::Path;
use std::time::Instant;

#[global_allocator]
static GLOBAL: jemallocator::Jemalloc = jemallocator::Jemalloc;

/// clap parser
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Path to HDF5 source
    source_path: String,
    output_path: String,
    partition_size: u32,
    data_set_label: String,
}

const NUM_NEIGHBOURS: usize = 18;
const DELTA: f64 = 0.01;
const REVERSE_LIST_SIZE: usize = 32;
const NUM_VERTICES: usize = 333;

pub fn main() -> Result<()> {
    pretty_env_logger::formatted_timed_builder()
        .filter_level(log::LevelFilter::Trace)
        .init();

    let args = Args::parse();

    log::info!("Loading h5 data files...");

    let dir_base = Path::new(&args.source_path);
    if !dir_base.is_dir() {
        anyhow::bail!("{} is not a directory", args.source_path);
    }

    let file_names = get_file_names(dir_base, "img_emb_", ".h5").unwrap();
    let sizes = get_file_sizes(dir_base, &file_names, &args.data_set_label).unwrap();
    let partition_boundaries = partition_into(&sizes, args.partition_size).unwrap(); // 2_200_000
    let partitions = make_partitions(&partition_boundaries, &file_names);

    let mut start_index = 0;

    for i in 0..partitions.len() {
        let vec = &partitions[i];

        let part = partitions.get(i).unwrap();

        let dao = load_h5_files::<Simd256x2, { DATA_DIM }>(
            dir_base,
            part,
            NUM_VERTICES,
            start_index,
            &args.data_set_label,
        )
        .unwrap();

        log::info!(
            "Loaded partition: {} from: {:?} Dao base: {} size = {}",
            i,
            vec,
            dao.base_addr,
            dao.num_data
        );

        let partition_data_size = dao.embeddings.len() as u32;

        let file_name = "nn_table".to_string().add(i.to_string().as_str());

        create_and_store_nn_table(
            dao,
            NUM_NEIGHBOURS,
            REVERSE_LIST_SIZE,
            DELTA,
            start_index,
            &args.output_path,
            &file_name,
        );

        start_index = start_index + partition_data_size;
    }

    Ok(())
}
