use big_knn::NalityNNTable;
use big_knn::{get_row_iterator, RowIter};
use bincode;
use clap::Parser;
use serde::Deserialize;
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use utils::address::GlobalAddress;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Path to HDF5 source
    file_path: String,
    num_rows: usize,
}

fn main() {
    let args = Args::parse();
    let nn_table_path = &args.file_path;
    let num_rows = args.num_rows;

    log::info!("Loading json NN table from {:?}", nn_table_path);

    let iter = get_row_iterator(nn_table_path);

    let mut i = 0;

    iter.take(num_rows).for_each(|row| {
        print!("{i}: ");
        for n in row.iter() {
            print!("id: {:?} sim: {:?}  ", n.id().as_usize(), n.sim());
        }
        println!();
        i = i + 1;
    });
}
