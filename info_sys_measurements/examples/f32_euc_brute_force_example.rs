use anyhow::Result;
use ndarray::{Array1, ArrayView1, s};
use rand::random;
use std::time::Instant;

fn main() -> Result<()> {
    let num_data = 1_000_000;

    for dims in [100, 384, 500, 768] {
        do_experiment(num_data, dims)
    }

    Ok(())
}

fn do_experiment(num_data: usize, dims: usize) {
    let query = Array1::from_iter((0..dims).map(|_| random::<f32>()));
    let data = Array1::from_iter((0..dims * num_data).map(|_| random::<f32>()));

    let now = Instant::now();

    // Do a brute force of queries against the data

    let distances = generate_euc_dists(query, data, num_data, dims);

    let after = Instant::now();

    eprintln!("Last distance is {:?}", distances.iter().last());

    println!(
        "F32 {}:\t{} ns",
        dims,
        ((after - now).as_nanos() as f64) as f64
    );
}

pub fn euc_view(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> f32 {
    // f32::sqrt(a.iter().zip(b.iter()).map(|(a, b)| (a - b).powi(2)).sum())
    a.iter()
        .zip(b.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt()
}

fn generate_euc_dists(
    query: Array1<f32>,
    data: Array1<f32>,
    num_data: usize,
    dims: usize,
) -> Vec<f32> {
    let q = query.slice(s![0..dims]);
    (0..num_data)
        .map(|data_index| {
            let d = data.slice(s![data_index * dims..(data_index * dims) + dims]);
            euc_view(&q, &d)
        })
        .collect::<Vec<f32>>()
}
