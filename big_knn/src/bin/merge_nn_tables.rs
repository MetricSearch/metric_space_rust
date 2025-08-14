/*
  Program to merge NN tables from Laion-400 h5 files.
*/

use anyhow::bail;
use big_knn::big_knn_r_descent::make_big_knn_table2_bsp;
use big_knn::dao_manager::{DaoManager, DaoStore};
use big_knn::{get_file_names, get_partitions, write_table, NalityNNTable, DATA_DIM};
use bits::container::{BitsContainer, Simd256x2};
use bits::EvpBits;
use clap::Parser;
use dao::hdf5_to_dao_loader::load_h5_files;
use dao::Dao;
use itertools::Itertools;
use ndarray::{concatenate, s, stack, Array2, ArrayView2, Axis, ShapeError, Zip};
use r_descent::{only_initialise_table_bsp_randomly, IntoRDescent, RDescent};
use std::env::args;
use std::fs::File;
use std::io::{BufReader, BufWriter, Seek};
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
    partition_size: u32,
    data_set_label: String,
}

pub fn main() -> anyhow::Result<()> {
    pretty_env_logger::formatted_timed_builder()
        .filter_level(log::LevelFilter::Trace)
        .init();

    let start_time = Instant::now();
    log::info!("Establishing Source NN tables ...");

    let args = Args::parse();

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

    let embeddings_path = Path::new(&args.raw_data_source_path);
    if !embeddings_path.is_dir() {
        anyhow::bail!("{} is not a directory", args.raw_data_source_path);
    }

    let (_, partitions) =
        get_partitions(embeddings_path, args.partition_size, &args.data_set_label);

    let h5_file_names = get_file_names(embeddings_path, "img_emb_", ".h5").unwrap();

    let h5_sizes: Vec<usize> = h5_file_names // sizes of each embeddings data file
        .iter()
        .map(|fname| {
            let path = embeddings_path.join(&fname);
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
        // the names of the h5 files that contain the raw data

        let first_part_file_names = &partitions[pair[0].0];
        let second_part_file_names = &partitions[pair[1].0];

        log::info!(
            "Building NN table using data files from partition: {:?} and partition: {:?}",
            first_part_file_names,
            second_part_file_names
        );

        let mut base_address = 0;

        if let Some((_, start_addr)) = h5_file_names_and_starts // TODO look at this mess!!!!!
            .iter()
            .find(|(fname, _)| *fname == &first_part_file_names[0])
        {
            base_address = **start_addr;
        } else {
            bail!("Cannot find file {} in h5 files", &first_part_file_names[0]);
        }

        log::info!("Base address of part 1 is: {}", base_address);

        let dao1: Dao<EvpBits<Simd256x2, { DATA_DIM }>> = load_h5_files::<Simd256x2, { DATA_DIM }>(
            embeddings_path,
            first_part_file_names,
            NUM_VERTICES,
            base_address as u32, // the base address of the first h5 file in the part
            &args.data_set_label,
        )
        .unwrap();

        let part1_size = dao1.num_data; // TODO we need the start indices of each

        if let Some((_, start_addr)) = h5_file_names_and_starts
            .iter()
            .find(|(fname, _)| *fname == &second_part_file_names[0])
        {
            base_address = **start_addr;
        } else {
            bail!(
                "Cannot find file {} in h5 files",
                &second_part_file_names[0],
            );
        }

        log::info!("Base address of part 2 is: {}", base_address);

        let dao2: Dao<EvpBits<Simd256x2, { DATA_DIM }>> = load_h5_files::<Simd256x2, { DATA_DIM }>(
            embeddings_path,
            second_part_file_names,
            NUM_VERTICES,
            base_address as u32, // the base address of the first h5 file in the part
            &args.data_set_label,
        )
        .unwrap();

        let part2_size = dao2.num_data;

        let mut daos: Vec<Dao<EvpBits<Simd256x2, { DATA_DIM }>>> = vec![];
        daos.push(dao1);
        daos.push(dao2);

        for dao in &daos {
            println!(
                "Loading dao range: [{}..{}] (inc)",
                dao.base_addr,
                dao.base_addr as usize + dao.num_data - 1
            );
        }

        // Now get the NN tables

        let mut first_nn_table_path =
            Path::new(&args.nn_tables_source_dir).join("nn_table".to_string().add(pair[0].1));
        first_nn_table_path.set_extension("bin");

        let mut second_nn_table_path =
            Path::new(&args.nn_tables_source_dir).join("nn_table".to_string().add(pair[1].1));
        second_nn_table_path.set_extension("bin");

        let combined_nn_table = combine_nn_table(&first_nn_table_path, &second_nn_table_path, daos);

        split_and_write_back(
            combined_nn_table,
            first_nn_table_path,
            part1_size,
            second_nn_table_path,
            part2_size,
        );
    }

    let end = Instant::now();

    let final_time = Instant::now();
    log::trace!(
        "Time To merge all NN tables: {} ms",
        ((final_time - start_time).as_millis() as f64)
    );

    Ok(())
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
    nn_table: NalityNNTable,
    nn_table1_path: PathBuf,
    part1_size: usize,
    nn_table2_path: PathBuf,
    part2_size: usize,
) {
    let mut nalities = nn_table.nalities;

    let original_table_width = nalities.shape()[1] / 2;
    println!("Splitting best {}", original_table_width);

    let nalities = sort_table(nalities);

    let top_nalities: ArrayView2<_> = nalities.slice(s![0..part1_size, 0..original_table_width]);
    let bottom_nalities: ArrayView2<_> = nalities.slice(s![part2_size.., 0..original_table_width]);

    let nn_table_1 = NalityNNTable {
        // TODO More copying???
        nalities: top_nalities.to_owned(),
    };

    write_table(&nn_table1_path, &nn_table_1);

    let nn_table_2 = NalityNNTable {
        // TODO More copying???
        nalities: bottom_nalities.to_owned(),
    };

    write_table(&nn_table2_path, &nn_table_2);
}

fn combine_nn_table(
    nn_table1_path: &PathBuf,
    nn_table2_path: &PathBuf,
    daos: Vec<Dao<EvpBits<Simd256x2, { DATA_DIM }>>>,
) -> NalityNNTable {
    let nn_table1 = get_nn_table(&nn_table1_path);
    let nn_table2 = get_nn_table(&nn_table2_path);

    let combined_nalities: Array2<Nality> = concatenate(
        // Does this make a copy?
        Axis(0),
        &[nn_table1.nalities.view(), nn_table2.nalities.view()],
    )
    .unwrap();

    let dao_manager = DaoStore::new(daos);

    let loaded_daos = &dao_manager.daos;
    let num_neighbours = nn_table1.nalities.ncols();

    // let mapped_daos = dao_manager.daos;
    let glue_0 = only_initialise_table_bsp_randomly(
        dao_manager.daos[0].num_data,
        num_neighbours,
        dao_manager.daos[0].base_addr,
    );

    let glue_1 = only_initialise_table_bsp_randomly(
        dao_manager.daos[1].num_data,
        num_neighbours,
        dao_manager.daos[1].base_addr,
    );

    // TODO look at copy-free. Ferdia!
    let glue = concatenate![Axis(0), glue_1.view(), glue_0.view()];

    let combined_nalities = concatenate![Axis(1), combined_nalities.view(), glue.view()];

    let num_neighbours = num_neighbours * 2;

    make_big_knn_table2_bsp(
        dao_manager,
        combined_nalities.nrows(),
        &combined_nalities,
        combined_nalities.ncols(),
        DELTA,             // TODO hard code for the minute fix later
        REVERSE_LIST_SIZE, // TODO hard code for the minute fix later
    );

    NalityNNTable {
        nalities: combined_nalities,
    }
}

fn get_nn_table(nn_table_path: &PathBuf) -> NalityNNTable {
    println!("Loading NN table from {:?}", nn_table_path);
    let file = File::open(nn_table_path).unwrap();
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
