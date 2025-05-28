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
use r_descent_matrix::{get_nn_table2_bsp, initialise_table_bsp};
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
    let (mut ords, mut dists) = initialise_table_bsp(dao_bsp.clone(), chunk_size, num_neighbours);

    // log::info!("ORDS: {:?}", ords);
    // log::info!("Dists: {:?}", dists);
    //
    // for i in 0..3 {
    //     log::info!("Row {} ids: {:?} dists: {:?} ", i, ords.row(i), dists.row(i));
    //     for ord in ords.row(i) {
    //         log::info!( "dist({},{}) real dist is: {} ", i, ord, bsp_similarity_as_f32::<2>(dao_bsp.get_datum(i), dao_bsp.get_datum(*ord)) );
    //     }
    // }

    log::info!("Getting NN table");

    get_nn_table2_bsp(
        dao_bsp.clone(),
        &mut ords,
        &mut dists,
        num_neighbours,
        rho,
        delta,
        reverse_list_size,
    );

    log::info!("Line 0 of table:");
    for i in 0..10 {
        log::info!(" neighbours: {} dists: {}", ords[[0, i]], dists[[0, i]]);
    }

    let end = Instant::now();

    log::info!(
        "Finished (including load time in {} s",
        (end - start).as_secs()
    );
    log::info!(
        "Finished (post load time) in {} s",
        (end - start_post_load).as_secs()
    );

    Ok(())
}
