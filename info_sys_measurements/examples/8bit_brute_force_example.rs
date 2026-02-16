use anyhow::Result;
use ndarray::{Array1, ArrayView1, s};
use rand::random;
use rayon::prelude::*;
use std::time::Instant;

pub fn euc_8bit(a: &ArrayView1<i8>, b: &ArrayView1<i8>) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(a, b)| (a - b).pow(2) as f32)
        .sum()
}

pub fn get_max(data: &Array1<f32>) -> f32 {
    return data.iter().cloned().reduce(f32::max).unwrap();
}

pub fn to_i8_array(array: &Array1<f32>, max_f32: f32) -> Array1<i8> {
    array.mapv(|x| {
        let value = x / max_f32;

        if value.is_nan() {
            // this will never happen
            0
        } else {
            (value * i8::MAX as f32)
                .round()
                .clamp(i8::MIN as f32, i8::MAX as f32) as i8
        }
    })
}

fn main() -> Result<()> {
    let num_queries = 100;
    let num_data = 1_000_000;

    for dims in [100, 384, 500, 768] {
        do_experiment(num_queries, num_data, dims)
    }

    Ok(())
}

fn do_experiment(num_queries: usize, num_data: usize, dims: usize) {
    let queries_f32 = Array1::from_iter((0..dims * num_queries).map(|_| random::<f32>()));
    let data_f32 = Array1::from_iter((0..dims * num_data).map(|_| random::<f32>()));

    let max_f32 = data_f32.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    let queries = to_i8_array(&queries_f32, max_f32);
    let data = to_i8_array(&data_f32, max_f32);

    let now = Instant::now();

    // Do a brute force of queries against the data

    let eight_bit_distances = generate_8bit_dists(queries, data, num_queries, num_data, dims);

    let after = Instant::now();

    println!(
        "Time per 8bit {} dim query 1_000_000 dists: {} ns",
        dims,
        ((after - now).as_nanos() as f64) / num_queries as f64
    );
}

fn generate_8bit_dists(
    queries: Array1<i8>,
    data: Array1<i8>,
    num_queries: usize,
    num_data: usize,
    dims: usize,
) -> Vec<Vec<f32>> {
    (0..num_queries)
        .map(|q_index| {
            (0..num_data)
                .map(|data_index| {
                    let q = queries.slice(s![q_index * dims..(q_index * dims) + dims]);
                    let d = data.slice(s![data_index * dims..(data_index * dims) + dims]);
                    euc_8bit(&q, &d)
                })
                .collect::<Vec<f32>>()
        })
        .collect::<Vec<Vec<f32>>>()
}
