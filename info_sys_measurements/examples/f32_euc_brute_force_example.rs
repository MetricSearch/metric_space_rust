use anyhow::Result;
use ndarray::{Array1, ArrayView1, s};
use rand::random;
use rayon::iter::{ParallelBridge, ParallelIterator};
use std::time::Instant;

fn main() -> Result<()> {
    let num_queries = 100;
    let num_data = 1_000_000;

    for dims in [100, 384, 500, 768] {
        do_experiment(num_queries, num_data, dims)
    }

    Ok(())
}

fn do_experiment(num_queries: usize, num_data: usize, dims: usize) {
    let queries = Array1::from_iter((0..dims * num_queries).map(|_| random::<f32>()));
    let data = Array1::from_iter((0..dims * num_data).map(|_| random::<f32>()));

    let now = Instant::now();

    // Do a brute force of queries against the data

    let distances = generate_euc_dists(queries, data, num_queries, num_data, dims);

    let after = Instant::now();

    println!("Sum of distances is {:?}", distances.iter().flatten().sum::<f32>());

    println!(
        "Time per pca {} dim query 1_000_000 dists: {} ns",
        dims,
        ((after - now).as_nanos() as f64) / num_queries as f64
    );
}

pub fn euc_view(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> f32 {
    // f32::sqrt(a.iter().zip(b.iter()).map(|(a, b)| (a - b).powi(2)).sum())
    a.iter().zip(b.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f32>().sqrt()
}

fn generate_euc_dists(
    queries: Array1<f32>,
    data: Array1<f32>,
    num_queries: usize,
    num_data: usize,
    dims: usize,
) -> Vec<Vec<f32>> {
    (0..num_queries)
        .par_bridge()
        .map(|q_index| {
            (0..num_data)
                .map(|data_index| {
                    let q = queries.slice(s![q_index * dims..(q_index * dims) + dims]);
                    let d = data.slice(s![data_index * dims..(data_index * dims) + dims]);
                    euc_view(&q, &d)
                })
                .collect::<Vec<f32>>()
        })
        .collect::<Vec<Vec<f32>>>()
}
