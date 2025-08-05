use anyhow::Result;
use bits::container::Simd256x2;
use bits::EvpBits;
use clap::Parser;
use dao::csv_dao_loader::dao_from_csv_dir;
use dao::hdf5_to_dao_loader::hdf5_f32_to_bsp_load;
use dao::pubmed_hdf5_gt_loader::hdf5_pubmed_gt_load;
use deepsize::DeepSizeOf;
use r_descent::{IntoRDescent, RDescent};
use std::rc::Rc;
use std::time::Instant;
use utils::bytes_fmt;

/// clap parser
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Path to HDF5
    path: String,
}

fn main() -> Result<()> {
    pretty_env_logger::formatted_timed_builder()
        .filter_level(log::LevelFilter::Trace)
        .init();

    let args = Args::parse();

    log::info!("Loading Pubmed data...");
    //let start = Instant::now();

    let num_queries = 10_000;
    const ALL_RECORDS: usize = 0;
    const NUM_VERTICES: usize = 200;

    let dao_bsp = Rc::new(
        hdf5_f32_to_bsp_load::<Simd256x2, 384>(&args.path, ALL_RECORDS, num_queries, NUM_VERTICES)
            .unwrap(),
    );

    log::info!("DAO BSP size {}", bytes_fmt(dao_bsp.deep_size_of()));

    let queries = dao_bsp.get_queries();
    let data = dao_bsp.get_data();

    log::info!(
        "Pubmed data size: {} queries size: {}, num data: {}",
        data.len(),
        queries.len(),
        dao_bsp.num_data,
    );

    let start_post_load = Instant::now();

    let num_neighbours = 10;
    let chunk_size = 200;
    let delta = 0.01;
    let reverse_list_size = 32;

    log::info!("Initializing NN table with chunk size {}", chunk_size);

    let start = Instant::now();

    let descent: RDescent = dao_bsp
        .clone()
        .into_rdescent(num_neighbours, reverse_list_size, delta);

    log::info!("Line 0 of table:");
    for i in 0..10 {
        log::info!(
            " neighbours: {} dists: {}",
            descent.neighbours[[0, i]],
            descent.similarities[[0, i]]
        );
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
