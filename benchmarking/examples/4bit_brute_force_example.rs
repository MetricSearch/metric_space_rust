use benchmarking::distances::{euc_4bit, to_packed_4bit_vectors, to_u4_array};
use anyhow::Result;
use ndarray::Array1;
use rand::random;
use std::time::Instant;
use wide::u8x32;

fn main() -> Result<()> {
    let num_data = 1_000_000;

    //----------------

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

    let query = to_u4_array(&query_f32, max_f32);
    let data = to_u4_array(&data_f32, max_f32);

    let query_wide = to_packed_4bit_vectors(&query, dims);
    let data_wide = to_packed_4bit_vectors(&data, dims);

    let now = Instant::now();

    // Do a brute force of queries against the data
    let four_bit_distances = generate_4bit_dists(&query_wide, &data_wide, dims);

    let after = Instant::now();
    eprintln!("Last distance is {:?}", four_bit_distances.iter().last());

    println!(
        "Int4 {}:\t{} ns",
        dims,
        ((after - now).as_nanos() as f64) as f64
    );
}

fn generate_4bit_dists(query: &Vec<u8x32>, data: &Vec<u8x32>, dims: usize) -> Vec<f32> {
    assert!(dims > 0);
    let blocks_per_vector = dims.div_ceil(64);
    assert_eq!(query.len(), blocks_per_vector);
    assert_eq!(data.len() % blocks_per_vector, 0);

    data.chunks_exact(blocks_per_vector)
        .map(|d| euc_4bit(query, d))
        .collect()
}

