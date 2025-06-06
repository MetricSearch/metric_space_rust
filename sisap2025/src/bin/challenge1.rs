// /*
// This task challenges participants to develop memory-efficient indexing solutions with reranking capabilities. Each solution will be run in a Linux container with limited memory and storage resources.
// Container specifications: 8 virtual CPUs, 16 GB of RAM, the dataset will be mounted read-only into the container.
// Wall clock time for the entire task: 12 hours.
// Minimum average recall to be considered in the final ranking: 0.7.
// Dataset: PUBMED23 (23 million vectors (384 dimensions) with out-of-distribution queries).
// The goal is to evaluate k=30 nearest neighbors for a large set of query objects, as follows:
// The final score of each team is measured as the best throughput evaluated on up to 16 different search hyperparameters.
// Teams are provided with a public set of 11,000 query objects for development purposes.
// A private set of 10,000 new queries will be used for the final evaluation.
//  */
// // Code originates from metric_space/r_descent/examples/check_r_descent_bsp_pubmed.rs

use anyhow::Result;
use bits::{bsp_distance_as_f32, EvpBits};
use clap::Parser;
use dao::hdf5_to_dao_loader::hdf5_f32_to_bsp_load;
use dao::Dao;
use hdf5::File as Hdf5File;
use ndarray::{s, Array2, ArrayView1};
use r_descent::{IntoRDescentWithRevNNs, RDescentWithRev, RevSearch};
use std::rc::Rc;
use std::time::Instant;
use utils::pair::Pair;

#[global_allocator]
static GLOBAL: jemallocator::Jemalloc = jemallocator::Jemalloc;

const NUM_NEIGHBOURS_IN_NN_TABLE: usize = 10;
const DELTA: f64 = 0.01;
const BUILD_REVERSE_LIST_SIZE: usize = 32;
const NUM_NEIGHBOURS_IN_REVERSE_TABLE: usize = 16;
const NUM_RESULTS_REQUIRED: usize = 30;

/// clap parser
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Path to HDF5
    source_path: String,
    /// Path to results file
    output_path: String,
}

fn main() -> Result<()> {
    pretty_env_logger::formatted_timed_builder()
        .filter_level(log::LevelFilter::Trace)
        .init();

    log::error!("{}", size_of::<EvpBits<2>>());

    let args = Args::parse();

    log::info!("Loading Pubmed data...");
    let start = Instant::now();

    let num_queries = 10_000;
    const ALL_RECORDS: usize = 0;
    const NUM_VERTICES: usize = 256;

    let dao_bsp: Rc<Dao<EvpBits<2>>> = Rc::new(
        hdf5_f32_to_bsp_load(&args.source_path, ALL_RECORDS, num_queries, NUM_VERTICES).unwrap(),
    );

    let queries: ArrayView1<EvpBits<2>> = dao_bsp.get_queries();

    let queries = queries.slice(s!(0..1000));

    let data: ArrayView1<EvpBits<2>> = dao_bsp.get_data();

    log::info!(
        "Pubmed data size: {} queries size: {}, num data: {}",
        data.len(),
        queries.len(),
        dao_bsp.num_data,
    );

    log::info!("Getting NN table");

    let descent = dao_bsp.clone().into_rdescent_with_rev_nn(
        NUM_NEIGHBOURS_IN_NN_TABLE,
        BUILD_REVERSE_LIST_SIZE,
        DELTA,
        NUM_NEIGHBOURS_IN_REVERSE_TABLE,
    );

    let end = Instant::now();

    log::info!(
        "Finished (including load time in {} s",
        (end - start).as_secs()
    );

    // let (gt_nns, _gt_dists) = hdf5_pubmed_gt_load(&args.path, knns).unwrap();
    // let gt_queries = dao_bsp.get_queries();

    log::info!("Pubmed Results:");

    println!("Doing {:?} queries", queries.len());

    do_queries(
        queries.to_vec(),
        &descent,
        dao_bsp.clone(),
        bsp_distance_as_f32,
        args.output_path,
        NUM_RESULTS_REQUIRED,
    );

    Ok(())
}

fn do_queries(
    queries: Vec<EvpBits<2>>,
    descent: &RDescentWithRev,
    dao: Rc<Dao<EvpBits<2>>>,
    distance: fn(&EvpBits<2>, &EvpBits<2>) -> f32,
    output_path: String,
    num_results: usize,
) {
    let start = Instant::now();

    let mut results = vec![];

    queries.iter().enumerate().for_each(|(_qid, query)| {
        let qresults = descent.rev_search(query.clone(), dao.clone(), 100, distance);
        results.push(qresults.iter().map(|i| *i + 1).take(num_results).collect());
    });

    let end = Instant::now();

    log::info!("Queries run in {} s", (end - start).as_secs());

    log::info!("Writing to h5 file {}", output_path);
    save_results(results, output_path);
    println!("Data saved to h5 file");
}

fn save_results(results: Vec<Vec<usize>>, output_path: String) {
    log::info!("Writing to h5 file {}", output_path);

    // Get the results into an Array2
    let rows = results.len();
    let cols = results.first().map_or(0, |row| row.len());

    let results: Vec<usize> = results.into_iter().flatten().collect();
    let results = Array2::from_shape_vec((rows, cols), results).expect("Shape mismatch");

    let _ = save_to_h5(&output_path, results);
}

pub fn save_to_h5(f_name: &str, results: Array2<usize>) -> Result<()> {
    let file = Hdf5File::create(f_name)?; // open for writing
    let group = file.create_group("/results")?; // create a group
    let builder = group.new_dataset_builder();
    let _ds = builder.with_data(&results.to_owned()).create("results")?;
    file.flush()?;
    Ok(())
}

fn _show_gt(qid: usize, gt_data: ArrayView1<usize>) {
    print!(
        "GT pairs size {} first few GT results for q{}:\t",
        gt_data.len(),
        qid
    );
    gt_data.iter().take(5).for_each(|index| {
        print!("{}", index);
    });
    println!();
}

fn _intersection_size(results: &Vec<Pair>, gt_indices: ArrayView1<usize>) -> usize {
    results
        .iter()
        .filter_map(|pair| {
            if gt_indices.as_slice().unwrap().contains(&pair.index) {
                Some(1)
            } else {
                None
            }
        })
        .count()
}
