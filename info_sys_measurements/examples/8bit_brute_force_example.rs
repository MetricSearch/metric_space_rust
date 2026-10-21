use anyhow::Result;
use ndarray::Array1;
use rand::random;
use std::time::Instant;
use wide::{i16x16, i32x8, u8x16, u8x32};

pub fn euc_8bit(a: &[u8x32], b: &[u8x32]) -> f32 {
    let mut sum = i32x8::ZERO;

    for (a, b) in a.iter().zip(b.iter()) {
        for (a, b) in a
            .as_array_ref()
            .chunks_exact(16)
            .zip(b.as_array_ref().chunks_exact(16))
        {
            // Need to map into 
            let diff =
                i16x16::from(u8x16::from(a)) -
                i16x16::from(u8x16::from(b));

            sum += diff.dot(diff);
        }
    }

    (sum.reduce_add() as f32).sqrt()
}

fn to_u8x32_vectors(array: &Array1<u8>, dims: usize) -> Vec<u8x32> {
    assert!(dims > 0);
    assert_eq!(array.len() % dims, 0);

    array
        .as_slice()
        .expect("array must be contiguous")
        .chunks_exact(dims)
        .flat_map(|vector| vector.chunks(32).map(u8x32::from))
        .collect()
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

    let query_wide = to_u8x32_vectors(&query, dims);
    let data_wide = to_u8x32_vectors(&data, dims);

    let now = Instant::now();

    // Do a brute force of queries against the data
    let eight_bit_distances = generate_8bit_dists(&query_wide, &data_wide, dims);

    let after = Instant::now();

    eprintln!("Last distance is {:?}", eight_bit_distances.iter().last());

    println!(
        "Int8 {}:\t{} ns",
        dims,
        ((after - now).as_nanos() as f64) as f64
    );
}

fn generate_8bit_dists(query: &[u8x32], data: &[u8x32], dims: usize) -> Vec<f32> {
    assert!(dims > 0);
    let blocks_per_vector = dims.div_ceil(32);
    assert_eq!(query.len(), blocks_per_vector);
    assert_eq!(data.len() % blocks_per_vector, 0);

    data.chunks_exact(blocks_per_vector)
        .map(|d| euc_8bit(query, d))
        .collect()
}