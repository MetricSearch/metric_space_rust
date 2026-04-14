/* A copy of challenge1 to test the Hamiltonians */

use anyhow::Result;
use clap::Parser;
use ndarray::{Array1, ArrayView, Ix1};

use half::f16;
use hamiltonians::{get_cycle_lengths, get_cycle_lookup_table, get_vertex_number, make_pascal};
use std::time::Instant;
use utils::arg_sort_big_to_small_2d;
use utils::non_nan::NonNan;

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

    log::info!("Loading Wikipedia data...");
    let start = Instant::now();

    const ALL_RECORDS: usize = 0;
    const NUM_QUERIES: usize = 0;
    const CHUNK_SIZE: usize = 8192;
    const D: usize = 1024;
    const NON_ZEROS: usize = 512;

    let data_f16: Vec<Array1<f16>> = dao::generic_loader::par_load::<_, half::f16, _, _>(
        &args.source_path,
        "train",
        None,
        CHUNK_SIZE,
        |embedding| embedding.mapv(|f| f),
    )
    .unwrap();

    let end = Instant::now();

    log::info!(
        "Wikipedia Loaded {} data in {} s",
        data_f16.len(),
        (end - start).as_secs()
    );

    // Build Pascal triangle
    let pas_tri: Vec<Vec<f64>> = make_pascal(D);

    // Build cycle lengths and lookup tables
    let c_lengths: Vec<usize> = get_cycle_lengths(NON_ZEROS);
    let mut tables: Vec<Vec<Vec<bool>>> = Vec::with_capacity(NON_ZEROS);
    for xi in 1..=NON_ZEROS {
        tables.push(get_cycle_lookup_table(c_lengths[xi - 1], xi, &pas_tri));
    }

    let F16_ZERO: f16 = f16::from_f32(0.0);

    for arrai in data_f16 {
        let vertex: Vec<bool> = f16_vec_to_bool_vec(arrai, F16_ZERO);
        let result: f64 = get_vertex_number(NON_ZEROS, D, &vertex, &c_lengths, &tables, &pas_tri);
        println!("Vertex number: {result}");
    }

    let end = Instant::now();

    log::info!(
        "Finished (including load time in {} s",
        (end - start).as_secs()
    );

    Ok(())
}

fn f16_vec_to_bool_vec(arrai: Array1<f16>, F16_ZERO: f16) -> Vec<bool> {
    let median = arrai.iter().sum();
    arrai.iter().map(|&x| x < F16_ZERO).collect()
}

pub fn arg_sort_big_to_small_1d(dists: Vec<f16>) -> (Vec<usize>, Vec<f32>) {
    let mut enumerated = dists
        .iter()
        .enumerate()
        .map(|(pos, dist)| (pos, dist.to_f32()))
        .collect::<Vec<(usize, &f32)>>(); // Vec of positions (ords) and values (dists as f32 for simplicity)
    enumerated.sort_by(|a, b| NonNan::new(*b.1).partial_cmp(&NonNan::new(*a.1)).unwrap());
    enumerated.into_iter().unzip()
}
