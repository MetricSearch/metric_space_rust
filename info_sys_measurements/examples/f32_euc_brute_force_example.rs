use anyhow::Result;
use ndarray::{Array1, ArrayView1, s};
use rand::random;
use std::time::Instant;

use wide::f32x8;

fn main() -> Result<()> {
    let num_data = 1_000_000;

    do_experiment::<100>(num_data);
    do_experiment::<384>(num_data);
    do_experiment::<500>(num_data);
    do_experiment::<768>(num_data);

    Ok(())
}

fn do_experiment<const DIMS: usize>(num_data: usize) {
    let query = Array1::from_iter((0..DIMS).map(|_| random::<f32>()));
    let data = Array1::from_iter((0..DIMS * num_data).map(|_| random::<f32>()));

    assert!(DIMS > 0);
    assert_eq!(data.len() % DIMS, 0);

    let query_wide: Vec<f32x8> = query
        .as_slice()
        .expect("query must be contiguous")
        .chunks(8)
        .map(f32x8::from)
        .collect();

    let data_wide: Vec<f32x8> = data
        .as_slice()
        .expect("data must be contiguous")
        .chunks_exact(DIMS)
        .flat_map(|vector| vector.chunks(8).map(f32x8::from))
        .collect();

    let blocks_per_vector = DIMS.div_ceil(8);

    let now = Instant::now();

    // Do a brute force of queries against the data

    let distances = generate_euc_dists::<DIMS>(&query_wide, &data_wide, blocks_per_vector);

    let after = Instant::now();

    eprintln!("Last distance is {:?}", distances.iter().last());

    println!(
        "F32 {}:\t{} ns",
        DIMS,
        ((after - now).as_nanos() as f64) as f64
    );
}

fn generate_euc_dists<const DIMS: usize>(query: &Vec<f32x8>, data: &Vec<f32x8>, blocks_per_vector: usize) -> Vec<f32> {
    data.chunks_exact(blocks_per_vector)
        .map(|d| euc_view(query, d))
        .collect()
}

fn euc_view(a: &[f32x8], b: &[f32x8]) -> f32 {
    let mut sum = f32x8::ZERO;

    for (&a, &b) in a.iter().zip(b.iter()) {
        let diff = a - b;
        sum += diff * diff;
    }

    sum.reduce_add().sqrt()
}
