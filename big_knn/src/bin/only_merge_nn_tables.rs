/*
  Program to merge NN tables from Laion-400 h5 files.
*/

use anyhow::bail;
use big_knn::big_knn_r_descent::{into_big_knn_r_descent, make_big_knn_table2_bsp};
use big_knn::dao_manager::{DaoManager, DaoStore};
use big_knn::{
    create_and_store_nn_table, get_file_names, get_file_sizes, get_partitions, make_partitions,
    partition_into, write_nalities, write_table, NalityNNTable, DATA_DIM,
};
use bits::container::{BitsContainer, Simd256x2};
use bits::EvpBits;
use clap::Parser;
use dao::hdf5_to_dao_loader::load_h5_files;
use dao::Dao;
use itertools::Itertools;
use ndarray::{concatenate, s, stack, Array2, ArrayView2, Axis, ShapeError, Zip};
use r_descent::{
    initialise_table_bsp_randomly_overwrite_row_0_with_coin_toss, IntoRDescent, RDescent,
};
use rand::distr::Alphanumeric;
use rand::Rng;
use std::env::args;
use std::fs;
use std::fs::File;
use std::io::{BufReader, BufWriter, Seek};
use std::ops::Add;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::time::Instant;
use utils::address::GlobalAddress;
use utils::Nality;

const NUM_NEIGHBOURS: usize = 18;
const NUM_VERTICES: usize = 333; // TODO this is copied - shouldn't be!
const DELTA: f64 = 0.01; // TODO this is copied - shouldn't be!
const REVERSE_LIST_SIZE: usize = 32; // TODO this is copied - shouldn't be!

/// clap parser
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    embeddings_path: String,
    nn_tables_dest_dir: String,
    partition_size: u32,
    data_set_label: String,
}

pub fn main() -> anyhow::Result<()> {
    pretty_env_logger::formatted_timed_builder()
        .filter_level(log::LevelFilter::Trace)
        .init();

    let start_time = Instant::now();

    let args = Args::parse();

    let nn_tables_dest_dir = &args.nn_tables_dest_dir;

    log::info!("Loading h5 data files...");

    let embeddings_dir_base = Path::new(&args.embeddings_path);
    if !embeddings_dir_base.is_dir() {
        anyhow::bail!("{} is not a directory", args.embeddings_path);
    }

    let file_names = get_file_names(embeddings_dir_base, "img_emb_", ".h5").unwrap();
    let sizes = get_file_sizes(embeddings_dir_base, &file_names, &args.data_set_label).unwrap();
    let partition_boundaries = partition_into(&sizes, args.partition_size).unwrap(); // 2_200_000
    let partitions = make_partitions(&partition_boundaries, &file_names);

    let mut part_names = vec![]; // indices of all the partitions as strings
    for i in 0..partitions.len() {
        part_names.push(i.to_string());
    }

    let h5_file_names = get_file_names(embeddings_dir_base, "img_emb_", ".h5").unwrap();

    let h5_sizes: Vec<usize> = h5_file_names // sizes of each embeddings data file
        .iter()
        .map(|fname| {
            let path = embeddings_dir_base.join(&fname);
            let file = hdf5::File::open(path).unwrap(); // open for reading
            let h5_data = file.dataset(&args.data_set_label).unwrap();
            let data_size = h5_data.shape()[0];
            data_size as usize
        })
        .collect();

    // the starting index of each of the h5 files.
    let h5_starts = {
        let mut starts = Vec::with_capacity(h5_sizes.len() + 1);
        starts.push(0); // the first file starts at zero
        for size in &h5_sizes {
            let last = *starts.last().unwrap();
            starts.push(last + size);
        }
        starts
    };

    let h5_file_names_and_starts = h5_file_names
        .iter()
        .zip(h5_starts.iter())
        .collect::<Vec<_>>();

    let mut part_names = vec![]; // indices of all the partitions as strings
    for i in 0..partitions.len() {
        part_names.push(i.to_string());
    }

    for pair in part_names.iter().enumerate().combinations(2) {
        let first_part_file_names = &partitions[pair[0].0];
        let second_part_file_names = &partitions[pair[1].0];

        create_nn_table_from_pair(
            nn_tables_dest_dir,
            first_part_file_names,
            second_part_file_names,
            &h5_file_names_and_starts,
            embeddings_dir_base,
            &args.data_set_label,
            pair,
        );
    }

    let final_time = Instant::now();
    log::trace!(
        "Time To create NN table: {} ms",
        ((final_time - start_time).as_millis() as f64)
    );

    Ok(())
}

fn create_nn_table_from_pair(
    nn_tables_dest_dir: &String,
    first_part_file_names: &Vec<String>,
    second_part_file_names: &Vec<String>,
    h5_file_names_and_starts: &Vec<(&String, &usize)>,
    embeddings_dir_base: &Path,
    data_set_label: &String,
    pair: Vec<(usize, &String)>,
) {
    log::info!(
        "Building NN table using data files from partition: {:?} and partition: {:?}",
        first_part_file_names,
        second_part_file_names
    );

    let (start_index1, dao1, part1_size) = get_dao(
        &first_part_file_names,
        &h5_file_names_and_starts,
        embeddings_dir_base,
        data_set_label,
    );
    let (start_index2, dao2, part2_size) = get_dao(
        &second_part_file_names,
        &h5_file_names_and_starts,
        embeddings_dir_base,
        data_set_label,
    );

    let mut daos: Vec<Dao<EvpBits<Simd256x2, { DATA_DIM }>>> = vec![];
    daos.push(dao1);
    daos.push(dao2);

    let dao_manager = DaoStore::new(daos);

    // Now create the NN tables

    let neighbourlarities = initialise_table_bsp_randomly_overwrite_row_0_with_coin_toss(
        NUM_NEIGHBOURS,
        start_index1,
        part1_size,
        start_index2,
        part2_size,
    );

    make_big_knn_table2_bsp(
        dao_manager,
        part1_size + part2_size,
        &neighbourlarities,
        NUM_NEIGHBOURS,
        DELTA,
        REVERSE_LIST_SIZE,
    );

    let first_dir = Path::new(&nn_tables_dest_dir)
        .join("nn_table".to_string())
        .join(pair[0].1);

    check_dir_exists(&first_dir);

    let mut first_nn_table_path = first_dir.join(&random_file_name(8));
    first_nn_table_path.set_extension("bin");

    let second_dir = Path::new(&nn_tables_dest_dir)
        .join("nn_table".to_string())
        .join(pair[1].1);

    check_dir_exists(&second_dir);

    let mut second_nn_table_path = second_dir.join(&random_file_name(8));
    second_nn_table_path.set_extension("bin");

    split_and_write_back(
        neighbourlarities,
        first_nn_table_path,
        part1_size,
        second_nn_table_path,
        part2_size,
    );
}

fn check_dir_exists(path: &Path) {
    if !path.exists() || !path.is_dir() {
        match fs::create_dir_all(path) {
            Ok(_) => log::trace!("Created directory: {}", path.display()),
            Err(_) => panic!("Directory for path {} could not be created", path.display()),
        }
    } else if !path.is_dir() {
        panic!(
            "Path {} already exists but is not a directory: ",
            path.display()
        );
    }
}

fn random_file_name(len: usize) -> String {
    let mut rng = rand::rng();
    let name: String = (0..len).map(|_| rng.sample(Alphanumeric) as char).collect();
    name
}

fn get_dao(
    first_part_file_names: &&Vec<String>,
    h5_file_names_and_starts: &&Vec<(&String, &usize)>,
    embeddings_dir_base: &Path,
    data_set_label: &String,
) -> (usize, Dao<EvpBits<Simd256x2, 384>>, usize) {
    let mut start_index = get_base_addr(&first_part_file_names, &h5_file_names_and_starts).unwrap();

    log::info!("Base address of partition is: {}", start_index);

    let dao1: Dao<EvpBits<Simd256x2, { DATA_DIM }>> = load_h5_files::<Simd256x2, { DATA_DIM }>(
        embeddings_dir_base,
        first_part_file_names,
        NUM_VERTICES,
        start_index as u32, // the base address of the first h5 file in the part
        data_set_label,
    )
    .unwrap();

    let part1_size = dao1.num_data;
    (start_index, dao1, part1_size)
}

fn get_base_addr(
    second_part_file_names: &Vec<String>,
    h5_file_names_and_starts: &Vec<(&String, &usize)>,
) -> anyhow::Result<usize> {
    if let Some((_, &start)) = h5_file_names_and_starts
        .iter()
        .find(|(fname, _)| *fname == &second_part_file_names[0])
    {
        Ok(start)
    } else {
        bail!(
            "Cannot find file {} in h5 files",
            &second_part_file_names[0],
        );
    }
}

// TODO make 0 copy!
fn sort_table(mut table: Array2<Nality>) -> Array2<Nality> {
    let mut cont_table = table.as_standard_layout();

    for mut row in cont_table.axis_iter_mut(ndarray::Axis(0)) {
        let slice: &mut [Nality] = row.as_slice_mut().unwrap();
        slice.sort_by(|a, b| b.sim().total_cmp(&a.sim()));
    }

    cont_table.to_owned()
}

/// Split up the combined table and write back to the orginal NN files.
fn split_and_write_back(
    nalities: Array2<Nality>,
    nn_table1_path: PathBuf,
    part1_size: usize,
    nn_table2_path: PathBuf,
    part2_size: usize,
) {
    let nalities = sort_table(nalities); // TODO Still copies - stop it.

    let top_nalities: ArrayView2<_> = nalities.slice(s![0..part1_size, 0..]);
    write_nalities(&nn_table1_path, &top_nalities);

    let bottom_nalities: ArrayView2<_> = nalities.slice(s![part2_size.., 0..]);
    write_nalities(&nn_table2_path, &bottom_nalities);
}
