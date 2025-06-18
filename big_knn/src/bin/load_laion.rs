/*
    First program to load data from Laion-400 h5 files.
    Load...
 */
use std::path::Path;
use anyhow::Result;
use bits::container::Simd256x2;
use bits::EvpBits;
use clap::Parser;
use dao::Dao;
use ndarray::{s, Array2, ArrayView1, ArrayView2};
use r_descent::IntoRDescent;
use std::rc::Rc;
use std::time::Instant;
use hdf5::File as Hdf5File;
use utils::arg_sort_big_to_small_2d;
use std::fs;
use big_knn::{load_chunks,get_file_sizes};


#[global_allocator]
static GLOBAL: jemallocator::Jemalloc = jemallocator::Jemalloc;

/// clap parser
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Path to HDF5 source
    source_path: String,
    // Path to HDF5 target
    // output_path: String,
}

pub fn main() -> Result<()> {
    pretty_env_logger::formatted_timed_builder()
        .filter_level(log::LevelFilter::Trace)
        .init();

    let args = Args::parse();

    log::info!("Loading Laion-400M data...");
    let start = Instant::now();

    let dir_base = Path::new(&args.source_path);
    if !dir_base.is_dir() {
        anyhow::bail!("{} is not a directory", args.source_path);
    }

    let file_names = get_file_names( dir_base ).unwrap();

    let sizes = get_file_sizes(dir_base, &file_names).unwrap();

    for i in 0..file_names.len() {
        log::info!("File {} size {}", file_names.get(i).unwrap(), sizes.get(i).unwrap());
    }

    // let dao = load_chunks( dir_base, file_names, 1_000_000 ).unwrap();
    //
    // let data = dao.get_data();
    //
    // let num_data = data.len();
    //
    // log::info!(
    //     "Laion-400M data size: {} | num data: {}",
    //     data.len(),
    //     dao.num_data,
    // );
    //
    // let start_post_load = Instant::now();
    //
    // let num_neighbours = 18;
    // let chunk_size = 1000;
    // let delta = 0.01;
    // let reverse_list_size = 64;
    //
    // log::info!("Getting NN table");
    //
    // let descent =
    //     dao
    //         .clone()
    //         .into_rdescent(num_neighbours, reverse_list_size, chunk_size, delta);
    //
    // let end = Instant::now();
    //
    // log::info!(
    //     "Finished (including load time in {} s",
    //     (end - start).as_secs()
    // );
    // log::info!(
    //     "Finished (post load time) in {} s",
    //     (end - start_post_load).as_secs()
    // );
    //
    // let neighbours = &descent.neighbours;
    // let sims = &descent.similarities;
    //
    // // Add 1 to all elements (preserving shape)
    // let neighbours = neighbours.mapv(|x| x + 1);
    //
    // let (ords, _) = arg_sort_big_to_small_2d(&sims.view()); // sort the data
    //
    // let selected_neighbours: Vec<usize> = {
    //     let neighbours_ref = &neighbours; // to avoid capture of neighbours
    //
    //     ords.rows()
    //         .into_iter()
    //         .enumerate()
    //         .flat_map(|(row_index, ord_row)| {
    //             ord_row
    //                 .iter()
    //                 .map(move |&col_index| neighbours_ref[[row_index, col_index]])
    //                 .collect::<Vec<_>>() // to avoid capture of row.
    //         })
    //         .collect()
    // };
    //
    // let selected_neighbours =
    //     Array2::from_shape_vec((num_data, num_neighbours), selected_neighbours)
    //         .expect("Failed to create Array2 - indexing error");

    // let selected_neighbours = selected_neighbours.slice(s![.., 1..16]); // get the first 15 columns.
    //
    // log::info!("Writing to h5 file {}", &args.output_path);
    // save_to_h5(&args.output_path, selected_neighbours)?;
    //
    // println!("Data saved to h5 file");

    Ok(())
}

pub fn save_to_h5(f_name: &str, data: ArrayView2<usize>) -> Result<()> {
    let file = Hdf5File::create(f_name)?; // open for writing
    let group = file.create_group("/knns")?; // create a group
                                             // TODO do they need the dists too?
    let builder = group.new_dataset_builder();

    let _ds = builder.with_data(&data.to_owned()).create("results")?;

    file.flush()?;

    Ok(())
}

fn get_file_names<P: AsRef<Path>>(path: P) -> Result<Vec<String>> {
    let mut file_names = Vec::new();

    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_file() {
            if let Some(name) = path.file_name() {
                if let Some(name_str) = name.to_str() {
                    file_names.push(name_str.to_string());
                }
            }
        }
    }

    Ok(file_names)
}
