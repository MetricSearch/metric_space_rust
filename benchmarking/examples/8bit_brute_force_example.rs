use anyhow::Result;
use benchmarking::distances::{euc_8bit, to_u8_array, to_u8x32_vectors};
use ndarray::Array1;
use rand::random;
use std::time::Instant;
use wide::u8x32;

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
