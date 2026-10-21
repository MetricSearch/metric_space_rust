use anyhow::Result;
use ndarray::Array1;
use rand::random;
use std::time::Instant;
use wide::{f32x8, u16x16};

pub fn euc_16bit(a: &[u16x16], b: &[u16x16]) -> f32 {
    let mut sum = f32x8::ZERO;

    for (a, b) in a.iter().zip(b.iter()) {
        for (a, b) in a
            .as_array_ref()
            .chunks_exact(8)
            .zip(b.as_array_ref().chunks_exact(8))
        {
            // Widen before subtraction and squaring; u16 values fit exactly in f32.
            let a = f32x8::new(std::array::from_fn(|lane| f32::from(a[lane])));
            let b = f32x8::new(std::array::from_fn(|lane| f32::from(b[lane])));
            let diff = a - b;
            sum += diff * diff;
        }
    }

    sum.reduce_add().sqrt()
}

fn to_u16x16_vectors(array: &Array1<u16>, dims: usize) -> Vec<u16x16> {
    assert!(dims > 0);
    assert_eq!(array.len() % dims, 0);

    array
        .as_slice()
        .expect("array must be contiguous")
        .chunks_exact(dims)
        .flat_map(|vector| vector.chunks(16).map(u16x16::from))
        .collect()
}

pub fn to_u16_array(array: &Array1<f32>, max_f32: f32) -> Array1<u16> {
    array.mapv(|x| {
        let value = x / max_f32;

        if value.is_nan() {
            // this will never happen
            0
        } else {
            (value * u16::MAX as f32)
                .round()
                .clamp(u16::MIN as f32, u16::MAX as f32) as u16
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

    let query = to_u16_array(&query_f32, max_f32);
    let data = to_u16_array(&data_f32, max_f32);

    let query_wide = to_u16x16_vectors(&query, dims);
    let data_wide = to_u16x16_vectors(&data, dims);

    let now = Instant::now();

    // Do a brute force of queries against the data

    let sixteen_bit_distances = generate_16bit_dists(&query_wide, &data_wide, dims);

    let after = Instant::now();
    eprintln!("Last distance is {:?}", sixteen_bit_distances.iter().last());

    println!(
        "Int16 {}:\t{} ns",
        dims,
        ((after - now).as_nanos() as f64) as f64
    );
}

fn generate_16bit_dists(query: &[u16x16], data: &[u16x16], dims: usize) -> Vec<f32> {
    assert!(dims > 0);
    let blocks_per_vector = dims.div_ceil(16);
    assert_eq!(query.len(), blocks_per_vector);
    assert_eq!(data.len() % blocks_per_vector, 0);

    data.chunks_exact(blocks_per_vector)
        .map(|d| euc_16bit(query, d))
        .collect()
}
