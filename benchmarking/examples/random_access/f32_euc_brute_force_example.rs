use anyhow::Result;
use benchmarking::distances::euc_view;
use benchmarking::util::perm_system_time;
use ndarray::Array1;
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

    // How many f32x8s are needed to store a single vector
    let f32x8s_required_per_vec = DIMS.div_ceil(8);

    let perm = perm_system_time(num_data);

    let now = Instant::now();

    // Do a brute force of queries against the data

    let distances =
        generate_euc_dists::<DIMS>(&query_wide, &data_wide, &perm, f32x8s_required_per_vec);

    let after = Instant::now();

    eprintln!("Last distance is {:?}", distances.iter().last());

    println!(
        "F32 {}:\t{} ns",
        DIMS,
        ((after - now).as_nanos() as f64) as f64
    );
}

fn generate_euc_dists<const DIMS: usize>(
    query: &Vec<f32x8>,
    data: &Vec<f32x8>,
    perm: &[usize],
    blocks_per_vector: usize,
) -> Vec<f32> {
    let mut dists = vec![0.0; data.len() / blocks_per_vector];

    // Write sequentially, read randomly
    for (seq_id, rand_id) in perm.iter().enumerate() {
        let d = &data[rand_id * blocks_per_vector..(rand_id + 1) * blocks_per_vector];
        dists[seq_id] = euc_view(query, d);
    }

    dists
}
