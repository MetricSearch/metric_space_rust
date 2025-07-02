/*
  Program to merge NN tables from Laion-400 h5 files.
*/
use anyhow::bail;
use big_knn::{get_file_names, get_partitions, NalityNNTable};
use clap::Parser;
use dao::hdf5_to_dao_loader::load_h5_files;
use itertools::Itertools;
use ndarray::s;
use r_descent::RDescent;
use std::env::args;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::rc::Rc;
use std::time::Instant;
use utils::address::GlobalAddress;

const NUM_VERTICES: usize = 333; // TODO this is copied - shouldn't be!

/// clap parser
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Path to descent source
    descent_source_file: String,
}

pub fn main() -> anyhow::Result<()> {
    let start = Instant::now();
    log::info!("Loading Descent file  ...");

    let args = Args::parse();

    let file_path = Path::new(&args.descent_source_file);
    if !file_path.is_file() {
        bail!("{} is not a file", args.descent_source_file);
    }

    let file = File::open(file_path).unwrap();
    let reader = BufReader::new(file);
    let nn_table: NalityNNTable = bincode::deserialize_from(reader).unwrap(); // just the NN table not the data!

    let neighbours = nn_table.nalities;

    println!("Nalities shape: {:?}", neighbours.shape());

    println!("Nalities first 10 of row: ");
    neighbours
        .row(0)
        .slice(s![0..10])
        .iter()
        .for_each(|x| println!("{}, {}", GlobalAddress::as_u32(x.id()), x.sim()));

    Ok(())
}
