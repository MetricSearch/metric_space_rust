use anyhow::Result;
use ndarray::{Array1, ArrayView1, s};
use rand::random;
use rayon::prelude::*;
use std::time::Instant;

pub fn euc_16bit(a: &ArrayView1<i16>, b: &ArrayView1<i16>) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(a, b)| (a - b).pow(2) as f32)
        .sum::<f32>()
}

pub fn to_i16_array(array: &Array1<f32>, max_f32: f32) -> Array1<i16> {
    array.mapv(|x| {
        let value = x / max_f32;

        if value.is_nan() {
            // this will never happen
            0
        } else {
            (value * i16::MAX as f32)
                .round()
                .clamp(i16::MIN as f32, i16::MAX as f32) as i16
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

    let max_f32 = data_f32.iter().copied().map(|x| x.abs()).fold(f32::NEG_INFINITY, f32::max);

    let queries = to_i16_array(&queries_f32, max_f32);
    let data = to_i16_array(&data_f32, max_f32);

    let now = Instant::now();

    // Do a brute force of queries against the data

    let sixteen_bit_distances = generate_16bit_dists(queries, data, num_queries, num_data, dims);

    let after = Instant::now();
    println!("Last distance is {:?}", sixteen_bit_distances.iter().flatten().last());


    println!(
        "Time per 16bit {} dim query 1_000_000 dists: {} ns",
        dims,
        ((after - now).as_nanos() as f64) / num_queries as f64
    );
}

fn generate_16bit_dists(
    queries: Array1<i16>,
    data: Array1<i16>,
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
                    euc_16bit(&q, &d)
                })
                .collect::<Vec<f32>>()
        })
        .collect::<Vec<Vec<f32>>>()
}
