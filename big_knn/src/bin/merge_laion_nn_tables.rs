/*
  Program to merge NN tables from Laion-400 h5 files.
*/
use anyhow::bail;
use big_knn::{get_file_names, get_partitions};
use bits::container::Simd256x2;
use clap::Parser;
use dao::hdf5_to_dao_loader::load_h5_files;
use itertools::Itertools;
use r_descent::RDescent;
use std::env::args;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::rc::Rc;
use std::time::Instant;

const NUM_VERTICES: usize = 333; // TODO this is copied - shouldn't be!

/// clap parser
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Path to HDF5 source
    nn_tables_source_dir: String,
    raw_data_base_name: String,
    //output_path: String,
}

pub fn main() -> anyhow::Result<()> {
    let start = Instant::now();
    log::info!("Establishing Source NN tables ...");

    let args = Args::parse();

    let nn_tables_dir_base = Path::new(&args.nn_tables_source_dir);
    if !nn_tables_dir_base.is_dir() {
        bail!("{} is not a directory", args.nn_tables_source_dir);
    }
    let nn_file_names = get_file_names(&nn_tables_dir_base).unwrap();
    // check they are all of the expected type by extension
    for file_name in nn_file_names.iter() {
        if !file_name.ends_with(".bin") {
            bail!("{} is not a bin file", file_name);
        }
    }

    let raw_data_base_dir = Path::new(&args.raw_data_base_name);
    if !raw_data_base_dir.is_dir() {
        anyhow::bail!("{} is not a directory", args.raw_data_base_name);
    }

    let (sizes, partitions) = get_partitions(raw_data_base_dir, 2_500_000);

    if partitions.len() != nn_file_names.len() {
        bail!(
            "Parititions size does not match {} num nn tables {}",
            partitions.len(),
            nn_file_names.len()
        );
    }

    let mut names = vec![];
    for i in 0..20 {
        names.push(i.to_string());
    }

    for pair in names.iter().combinations(2) {
        let first_source_files = partitions.get(extract_nn_index(pair[0]).unwrap());
        let second_source_files = partitions.get(extract_nn_index(pair[1]).unwrap());

        println!("combs: {:?}", pair);

        // let combined_source_files = first_source_files.append(second_source_files);
        //
        // //let combined_nn_table = combine_nn_table(pair[0], pair[1], pair[3]); // all the indices are wrong - all need incremented
        //
        // // the combined_dao indices are ok - provided that we fix the nn references!
        //
        // let combined_dao = Rc::new(
        //     load_h5_files::<Simd256x2, 512>(raw_data_base_dir, combined_source_files, NUM_VERTICES)
        //         .unwrap(),
        // );

        // run descent on the table.
        // We need to make some absolute indices for all the sub tables!
    }

    Ok(())
}

fn extract_nn_index(filename: &str) -> Option<usize> {
    // Extract the number between "img_emb_" and ".h5"
    filename
        .strip_prefix("nn_table")?
        .strip_suffix(".bin")?
        .parse::<usize>()
        .ok()
}

pub fn get_table(nn_file_name: &String, directory_path: &Path) -> RDescent {
    let file = File::open(directory_path.join(nn_file_name)).unwrap();
    let reader = BufReader::new(file);
    let descent: RDescent = bincode::deserialize_from(reader).unwrap(); // just the NN table not the data!
    descent
}
