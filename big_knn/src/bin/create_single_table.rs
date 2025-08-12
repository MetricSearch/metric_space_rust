/*
  Program to merge NN tables from Laion-400 h5 files.
*/

use anyhow::bail;
use big_knn::{get_file_names, NalityNNTable};
use clap::Parser;
use r_descent::RDescent;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::ops::Add;
use std::path::{Path, PathBuf};
//use std::time::Instant;

/// clap parser
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Path to HDF5 source
    nn_tables_source_dir: String,
    result_table_path: String,
}

pub fn main() -> anyhow::Result<()> {
    pretty_env_logger::formatted_timed_builder()
        .filter_level(log::LevelFilter::Trace)
        .init();

    let args = Args::parse();

    //let start = Instant::now();
    log::info!(
        "Creating Single NN table from files in {}",
        &args.nn_tables_source_dir
    );

    let nn_tables_dir_base = Path::new(&args.nn_tables_source_dir);
    if !nn_tables_dir_base.is_dir() {
        bail!("{} is not a directory", args.nn_tables_source_dir);
    }
    let nn_file_names = get_file_names(&nn_tables_dir_base, "nn_table", ".bin").unwrap();

    // check they are all of the expected type by extension
    for file_name in nn_file_names.iter() {
        if !file_name.ends_with(".bin") {
            bail!(
                "{} from  dir: {:?} is not a bin file",
                file_name,
                nn_tables_dir_base,
            );
        }
    }

    let output_file: File = File::create(args.result_table_path).unwrap();
    let mut output_writer = BufWriter::new(output_file);

    for file_name in nn_file_names.iter() {
        let mut nn_table_path = Path::new(&args.nn_tables_source_dir).join(file_name);
        nn_table_path.set_extension("bin");

        copy_nn_table(&nn_table_path, &mut output_writer);
    }

    Ok(())
}

fn copy_nn_table(nn_table_path: &PathBuf, writer: &mut BufWriter<File>) {
    let nn_table = get_nn_table(&nn_table_path);
    write_existing_table(&nn_table, writer);
}

pub fn get_nn_table(nn_table_path: &PathBuf) -> NalityNNTable {
    log::info!("Loading NN table from {:?}", nn_table_path);
    let file = File::open(nn_table_path).unwrap();
    let reader = BufReader::new(file);
    let nn_table: NalityNNTable = bincode::deserialize_from(reader).unwrap();
    nn_table
}

pub fn write_existing_table(nn_table: &NalityNNTable, writer: &mut BufWriter<File>) {
    let result = bincode::serialize_into(writer, &nn_table);
    if result.is_err() {
        panic!("Fatal error saving NN table");
    } else {
        log::trace!("Successfully added to output file");
    }
}
