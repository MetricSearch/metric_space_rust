#![feature(path_add_extension)]
/*
  Program to merge NN tables from Laion-400 h5 files.
*/

use anyhow::bail;
use big_knn::big_knn_r_descent::make_big_knn_table2_bsp;
use big_knn::dao_manager::{DaoManager, DaoStore};
use big_knn::{get_file_names, get_partitions, write_table, NalityNNTable};
use bits::container::{BitsContainer, Simd256x2};
use bits::EvpBits;
use clap::Parser;
use dao::hdf5_to_dao_loader::load_h5_files;
use dao::Dao;
use itertools::Itertools;
use ndarray::{concatenate, s, stack, Array2, ArrayView2, Axis, ShapeError, Zip};
use r_descent::{IntoRDescent, RDescent};
use std::env::args;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::ops::Add;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::time::Instant;
use utils::address::GlobalAddress;
use utils::Nality;

const NUM_VERTICES: usize = 333; // TODO this is copied - shouldn't be!

const DELTA: f64 = 0.01; // TODO this is copied - shouldn't be!
const REVERSE_LIST_SIZE: usize = 32; // TODO this is copied - shouldn't be!

/// clap parser
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Path to HDF5 source
    nn_tables_source_dir: String,
    raw_data_source_path: String,
    output_path: String,
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

    let embeddings_path = Path::new(&args.raw_data_source_path);
    if !embeddings_path.is_dir() {
        anyhow::bail!("{} is not a directory", args.raw_data_source_path);
    }

    let (sizes, partitions) = get_partitions(embeddings_path, 2_500_000);

    if partitions.len() != nn_file_names.len() {
        bail!(
            "Parititions size does not match {} num nn tables {}",
            partitions.len(),
            nn_file_names.len()
        );
    }

    let mut names = vec![]; // indices of all the partitions as strings
    for i in 0..partitions.len() {
        names.push(i.to_string());
    }

    let h5_sizes: Vec<usize> = names // sizes of each data file
        .iter()
        .map(|fname| {
            let path = embeddings_path.join(&fname);
            let file = hdf5::File::open(path).unwrap(); // open for reading
            let h5_data = file.dataset("data").unwrap(); // the data // TODO make a parameter.
            let data_size = h5_data.shape()[0];
            data_size as usize
        })
        .collect();

    let h5_starts = {
        let mut starts = Vec::with_capacity(h5_sizes.len() + 1);
        starts.push(0); // the first file starts at zero
        starts.extend_from_slice(&h5_sizes);
        starts
    };

    for pair in names.iter().enumerate().combinations(2) {
        println!("Merging data files: {:?}", pair); // message terrible

        // the names of the h5 files that contain the raw data

        let first_part = partitions
            .get(extract_nn_index(pair[0].1).unwrap())
            .unwrap();
        let dao1: Dao<EvpBits<Simd256x2, 512>> = load_h5_files::<Simd256x2, 512>(
            embeddings_path,
            first_part,
            NUM_VERTICES,
            h5_starts[pair[0].0] as u32,
        )
        .unwrap();

        let part1_size = dao1.num_data; // TODO we need the start indices of each

        let second_part = partitions
            .get(extract_nn_index(pair[1].1).unwrap())
            .unwrap();
        let base_address2 = 0;
        let dao2: Dao<EvpBits<Simd256x2, 512>> = load_h5_files::<Simd256x2, 512>(
            embeddings_path,
            second_part,
            NUM_VERTICES,
            h5_starts[pair[1].0] as u32,
        )
        .unwrap();

        let part2_size = dao2.num_data;

        let mut daos: Vec<Dao<EvpBits<Simd256x2, 512>>> = vec![];
        daos.push(dao1);
        daos.push(dao2);

        // Now get the NN tables

        let first_nn_table_path = Path::new(&args.output_path)
            .join("nn_table".to_string().add(pair[0].1))
            .with_added_extension("bin");

        let second_nn_table_path = Path::new(&args.output_path)
            .join("nn_table".to_string().add(pair[1].1))
            .with_added_extension("bin");

        let combined_nn_table = combine_nn_table(&first_nn_table_path, &second_nn_table_path, daos);

        split_and_write_back(
            combined_nn_table,
            first_nn_table_path,
            part1_size,
            second_nn_table_path,
            part2_size,
        );
    }

    Ok(())
}

/// Split up the combined table and write back to the orginal NN files.
fn split_and_write_back(
    nn_table: NalityNNTable,
    nn_table1_path: PathBuf,
    part1_size: usize,
    nn_table2_path: PathBuf,
    part2_size: usize,
) {
    let nalities = nn_table.nalities;

    let top_nalities: ArrayView2<_> = nalities.slice(s![0..part1_size, ..]);
    let bottom_nalities: ArrayView2<_> = nalities.slice(s![part1_size.., ..]);

    let nn_table_1 = NalityNNTable {
        // TODO More copying???
        nalities: top_nalities.to_owned(),
    };

    write_table(nn_table1_path, &nn_table_1);

    let nn_table_2 = NalityNNTable {
        // TODO More copying???
        nalities: bottom_nalities.to_owned(),
    };

    write_table(nn_table2_path, &nn_table_2);
}

fn combine_nn_table(
    nn_table1_path: &PathBuf,
    nn_table2_path: &PathBuf,
    daos: Vec<Dao<EvpBits<Simd256x2, 512>>>,
) -> NalityNNTable {
    let nn_table1 = get_nn_table(&nn_table1_path);
    let nn_table2 = get_nn_table(&nn_table2_path);

    // TODO this makes a copy!

    let combined_indices: Array2<usize> = concatenate(
        Axis(0),
        &[nn_table1.neighbours.view(), nn_table2.neighbours.view()],
    )
    .unwrap();
    let combined_dists: Array2<f32> = concatenate(
        Axis(0),
        &[nn_table1.similarities.view(), nn_table2.similarities.view()],
    )
    .unwrap();

    // TODO Then this makes another copy!!

    let rows = combined_indices.nrows();
    let cols = combined_indices.ncols();

    let mut nalities: Array2<Nality> = unsafe { Array2::uninit([rows, cols]).assume_init() }; // or any suitable type

    Zip::from(&combined_indices)
        .and(&combined_dists)
        .and(&mut nalities)
        .for_each(|&index, &sim, nality| {
            *nality = Nality::new(sim, GlobalAddress::into(index as u32))
        });

    make_big_knn_table2_bsp(
        daos,
        rows,
        &nalities,
        cols,
        DELTA,             // TODO hard code for the minute fix later
        REVERSE_LIST_SIZE, // TODO hard code for the minute fix later
    );

    NalityNNTable { nalities: nalities }
}

fn get_nn_table(nn_table_path: &PathBuf) -> RDescent {
    let file = File::create(nn_table_path).unwrap();
    bincode::deserialize_from(file).unwrap()
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
