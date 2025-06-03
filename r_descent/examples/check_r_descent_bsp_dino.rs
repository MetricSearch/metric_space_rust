use anyhow::Result;
use bits::{bsp_similarity, bsp_similarity_as_f32, f32_data_to_cubic_bitrep, whamming_distance};
use bitvec_simd::BitVecSimd;
use clap::Parser;
use dao::convert_f32_to_bsp::f32_dao_to_bsp;
use dao::csv_dao_matrix_loader::dao_matrix_from_csv_dir;
use dao::{Dao, DaoMatrix};
use deepsize::DeepSizeOf;
use metrics::euc;
use ndarray::Array1;
use r_descent::{get_nn_table2_bsp, initialise_table_bsp, IntoRDescent};
use std::collections::HashSet;
use std::fs::File;
use std::io::BufReader;
use std::rc::Rc;
use std::time::Instant;
use utils::{bytes_fmt, dot_product_f32};

/// clap parser
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Path to dino2 CSV + TXT
    path: String,
}

fn main() -> Result<()> {
    pretty_env_logger::formatted_timed_builder()
        .filter_level(log::LevelFilter::Trace)
        .init();

    let args = Args::parse();

    log::info!("Loading mf dino data...");
    let num_queries = 10_000;
    let num_data = 1_000_000 - num_queries;

    let start = Instant::now();

    let dao_f32: Rc<DaoMatrix<f32>> =
        Rc::new(dao_matrix_from_csv_dir(&args.path, num_data, num_queries)?);

    log::info!(
        "Loaded {}, converting mf dino to bsp...",
        bytes_fmt(dao_f32.deep_size_of())
    );

    let dao_bsp = f32_dao_to_bsp::<2>(dao_f32.clone(), 200);
    log::info!(
        "Loaded {}, converting mf dino to bsp...",
        bytes_fmt(dao_bsp.deep_size_of())
    );

    log::info!("Running r_descent<bsp>...");

    let start_post_load = Instant::now();

    let num_neighbours = 10;
    let chunk_size = 100; // 20000;
    let rho = 1.0;
    let delta = 0.01;
    let reverse_list_size = 32;

    log::info!("Initializing NN table with chunk size {}", chunk_size);

    let descent =
        dao_bsp
            .clone()
            .into_rdescent(num_neighbours, reverse_list_size, chunk_size, delta);

    // log::info!("Line 0 of table:");
    // for i in 0..10 {
    //     log::info!(" neighbours: {} dists: {}", descent.neighbours[0,i], descent.similarities[0,i]);
    // }
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

    todo!(); // TODO <<<<<<<<<<<

    Ok(())
}
