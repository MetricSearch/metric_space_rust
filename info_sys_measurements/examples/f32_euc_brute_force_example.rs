use anyhow::Result;
use ndarray::{Array1, ArrayView1, s};
use rand::random;
use std::time::Instant;

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

    let now = Instant::now();

    // Do a brute force of queries against the data

    let distances = generate_euc_dists::<DIMS>(query, data);

    let after = Instant::now();

    eprintln!("Last distance is {:?}", distances.iter().last());

    println!(
        "F32 {}:\t{} ns",
        DIMS,
        ((after - now).as_nanos() as f64) as f64
    );
}

fn generate_euc_dists<const DIMS: usize>(query: Array1<f32>, data: Array1<f32>) -> Vec<f32> {
    let q = query.slice(s![0..DIMS]);
    data.exact_chunks(DIMS)
        .into_iter()
        .map(|d| euc_view(&q, &d))
        .collect::<Vec<f32>>()
}

fn euc_view(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> f32 {
    // f32::sqrt(a.iter().zip(b.iter()).map(|(a, b)| (a - b).powi(2)).sum())
    a.iter()
        .zip(b.iter())
        .map(|(a, b)| {
            let sub = a.algebraic_sub(*b);
            sub.algebraic_mul(sub)
        })
        .sum::<f32>()
        .sqrt()
}
