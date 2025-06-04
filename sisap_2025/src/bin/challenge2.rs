/*
In this task, participants are asked to develop memory-efficient indexing solutions that will be used to compute an approximation of the k-nearest neighbor graph for k=15. Each solution will be run in a Linux container with limited memory and storage resources.
Container specifications: 8 virtual CPUs, 16 GB of RAM, the dataset will be mounted read-only into the container.
Wall clock time for the entire task: 12 hours.
Minimum average recall to be considered in the final ranking: 0.8.
Dataset: GOOAQ (3 million vectors (384 dimensions) ).
The goal is to compute the k-nearest neighbor graph (without self-references), i.e., find the k-nearest neighbors using all objects in the dataset as queries.
We will measure graph’s quality as the recall against a provided gold standard and the full computation time (i.e., including preprocessing, indexing, and search, and postprocessing)
We provide a development dataset; the evaluation phase will use an undisclosed dataset of similar size computed with the same neural model.
*/
use anyhow::Result;
use bits::EvpBits;
use clap::Parser;
use dao::hdf5_dao_loader::add_str_attr;
use dao::hdf5_to_dao_loader::hdf5_f32_to_bsp_load;
use dao::Dao;
use hdf5::{Dataset, File as Hdf5File};
use ndarray::{s, Array1, Array2, ArrayView, ArrayView1, ArrayView2, Axis, Ix1};
use r_descent::IntoRDescent;
use std::fs::File;
use std::io::Write;
use std::rc::Rc;
use std::time::Instant;
use utils::pair::Pair;
use utils::{arg_sort_big_to_small_1d, arg_sort_big_to_small_2d};

/// clap parser
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Path to HDF5 source
    source_path: String,
    /// Path to HDF5 target
    output_path: String,
}

fn main() -> Result<()> {
    pretty_env_logger::formatted_timed_builder()
        .filter_level(log::LevelFilter::Trace)
        .init();

    let args = Args::parse();

    log::info!("Loading GOOAQ data...");
    let start = Instant::now();

    const ALL_RECORDS: usize = 0;
    const NUM_VERTICES: usize = 256;
    const NUM_QUERIES: usize = 0;

    let dao_bsp: Rc<Dao<EvpBits<2>>> = Rc::new(
        hdf5_f32_to_bsp_load(&args.source_path, ALL_RECORDS, NUM_QUERIES, NUM_VERTICES).unwrap(),
    );

    let data: ArrayView1<EvpBits<2>> = dao_bsp.get_data();

    let num_data = data.len();

    log::info!(
        "GOOAQ data size: {} | num data: {}",
        data.len(),
        dao_bsp.num_data,
    );

    let start_post_load = Instant::now();

    let num_neighbours = 18;
    let chunk_size = 1000;
    let delta = 0.01;
    let reverse_list_size = 64;

    log::info!("Getting NN table");

    let descent =
        dao_bsp
            .clone()
            .into_rdescent(num_neighbours, reverse_list_size, chunk_size, delta);

    let end = Instant::now();

    log::info!(
        "Finished (including load time in {} s",
        (end - start).as_secs()
    );
    log::info!(
        "Finished (post load time) in {} s",
        (end - start_post_load).as_secs()
    );

    let neighbours = &descent.neighbours;
    let sims = &descent.similarities;

    // Add 1 to all elements (preserving shape)
    let neighbours = neighbours.mapv(|x| x + 1);

    let (ords, _) = arg_sort_big_to_small_2d(&sims.view()); // sort the data

    let selected_neighbours: Vec<usize> = {
        let neighbours_ref = &neighbours; // to avoid capture of neighbours

        ords.rows()
            .into_iter()
            .enumerate()
            .flat_map(|(row_index, ord_row)| {
                ord_row
                    .iter()
                    .map(move |&col_index| neighbours_ref[[row_index, col_index]])
                    .collect::<Vec<_>>() // to avoid capture of row.
            })
            .collect()
    };

    let selected_neighbours =
        Array2::from_shape_vec((num_data, num_neighbours), selected_neighbours)
            .expect("Failed to create Array2 - indexing error");

    let selected_neighbours = selected_neighbours.slice(s![.., 1..16]); // get the first 15 columns.

    println!("====== Printing First 10 Rows ======");
    for i in 0..10 {
        println!(
            "{:?}",
            selected_neighbours
                .row(i)
                .slice(s![0..])
                .iter()
                .map(|x| *x)
                .collect::<Vec<usize>>()
        );
    }

    let f_name = "./_scratch/challenge2_results.h5";
    log::info!("Writing to h5 file {}", &args.output_path);
    save_to_h5(f_name, selected_neighbours)?;

    println!("Data saved to h5 file");

    Ok(())
}

pub fn save_to_h5(f_name: &str, data: ArrayView2<usize>) -> Result<()> {
    let file = Hdf5File::create(f_name)?; // open for writing
    let group = file.create_group("/knns")?; // create a group
                                             // TODO do they need the dists too?
    let builder = group.new_dataset_builder();

    let ds = builder.with_data(&data.to_owned()).create("results")?;

    file.flush()?;

    Ok(())
}
