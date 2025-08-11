use big_knn::NalityNNTable;
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
}

fn main() {
    let args = Args::parse();
    let data: NalityNNTable = get_nn_table(Path::new(&args.file_path));

    println!("First 5 rows:");
    for i in 0..5 {
        let row = data.nalities.row(i);
        println!("{i}: ");
        for n in row.iter() {
            print!("id: {:?} sim: {:?}  ", n.id().as_usize(), n.sim());
        }
        println!();
    }
}

pub fn get_nn_table(nn_table_path: &Path) -> NalityNNTable {
    log::info!("Loading NN table from {:?}", nn_table_path);
    let file = File::open(nn_table_path).unwrap();
    let reader = BufReader::new(file);
    let nn_table: NalityNNTable = bincode::deserialize_from(reader).unwrap();
    nn_table
}
