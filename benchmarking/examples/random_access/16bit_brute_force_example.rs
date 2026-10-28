use anyhow::Result;
use benchmarking::distances::{euc_16bit, to_u16_array, to_u16x16_vectors};
use benchmarking::util::perm_system_time;
use ndarray::Array1;
use rand::random;
use std::time::Instant;
use wide::u16x16;

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

    let query = to_u16_array(&query_f32, max_f32);
    let data = to_u16_array(&data_f32, max_f32);

    let query_wide = to_u16x16_vectors(&query, dims);
    let data_wide = to_u16x16_vectors(&data, dims);

    let perm = perm_system_time(num_data);

    let now = Instant::now();

    // Do a brute force of queries against the data

    let sixteen_bit_distances = generate_16bit_dists(&query_wide, &data_wide, &perm, dims);

    let after = Instant::now();
    eprintln!("Last distance is {:?}", sixteen_bit_distances.iter().last());

    println!(
        "Int16 {}:\t{} ns",
        dims,
        ((after - now).as_nanos() as f64) as f64
    );
}

fn generate_16bit_dists(
    query: &[u16x16],
    data: &[u16x16],
    perm: &[usize],
    dims: usize,
) -> Vec<f32> {
    assert!(dims > 0);
    let blocks_per_vector = dims.div_ceil(16);
    assert_eq!(query.len(), blocks_per_vector);
    assert_eq!(data.len() % blocks_per_vector, 0);

    let mut dists = vec![0.0; data.len() / blocks_per_vector];

    // Write sequentially, read randomly
    for (seq_id, rand_id) in perm.iter().enumerate() {
        let d = &data[rand_id * blocks_per_vector..(rand_id + 1) * blocks_per_vector];
        dists[seq_id] = euc_16bit(query, d);
    }

    dists
}
