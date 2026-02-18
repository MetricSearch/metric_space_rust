use anyhow::Result;
use ndarray::{Array1, ArrayView1, s};
use rand::random;
use std::time::Instant;

// Changed to sqrt
pub fn euc_8bit(a: &ArrayView1<u8>, b: &ArrayView1<u8>) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(a, b)| (a - b).pow(2) as f32)
        .sum::<f32>()
}

pub fn to_u8_array(array: &Array1<f32>, max_f32: f32) -> Array1<u8> {
    array.mapv(|x| {
        let value = x / max_f32;

        if value.is_nan() {
            // this will never happen
            0
        } else {
            (value * u8::MAX as f32)
                .round()
                .clamp(u8::MIN as f32, u8::MAX as f32) as u8
        }
    })
}

fn main() -> Result<()> {
    let num_data = 1_000_000;

    for dims in [100, 384, 500, 768] {
        do_experiment(num_data, dims)
    }

    Ok(())
}

fn do_experiment(num_data: usize, dims: usize) {
    let query_f32 = Array1::from_iter((0..dims).map(|_| random::<f32>()));
    let data_f32 = Array1::from_iter((0..dims * num_data).map(|_| random::<f32>()));

    let max_f32 = data_f32
        .iter()
        .copied()
        .map(|x| x.abs())
        .fold(f32::NEG_INFINITY, f32::max);

    let query = to_u8_array(&query_f32, max_f32);
    let data = to_u8_array(&data_f32, max_f32);

    let now = Instant::now();

    // Do a brute force of queries against the data
    let eight_bit_distances = generate_8bit_dists(query, data, num_data, dims);

    let after = Instant::now();

    eprintln!(
        "Last distance is {:?}",
        eight_bit_distances.iter().last()
    );

    println!(
        "Int8 {}:\t{} ns",
        dims,
        ((after - now).as_nanos() as f64) as f64
    );
}

fn generate_8bit_dists(
    query: Array1<u8>,
    data: Array1<u8>,
    num_data: usize,
    dims: usize,
) -> Vec<f32> {
    let q = query.slice(s![0..dims]);

    (0..num_data)
        .map(|data_index| {
            let d = data.slice(s![data_index * dims..(data_index * dims) + dims]);
            euc_8bit(&q, &d)
        })
        .collect::<Vec<f32>>()
}
