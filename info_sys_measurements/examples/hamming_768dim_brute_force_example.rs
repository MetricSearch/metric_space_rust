use anyhow::Result;
use bits::container::BitsContainer;
use bits::container::Simd128;
use ndarray::Array1;
use rand::random;
use rayon::prelude::*;
use std::time::Instant;

fn one_bit_similarity_128(a: &Simd128, b: &Simd128) -> u32 {
    // a.into_u64_iter()
    //     .zip(b.into_u64_iter())
    //     .map(|(x, y)| (x ^ y).count_ones())
    //     .sum::<u32>()

    a.xor(b).count_ones() as u32
}

fn to_one_bit_128(embedding: Array1<f32>) -> Simd128 {
    let mut bits = Simd128::default();
    (0..embedding.len()).for_each(|index| {
        if embedding[index] > 0.0 {
            bits.set_bit(index, true);
        }
    });
    bits
}

fn main() -> Result<()> {
    let num_queries = 100;
    let num_data = 1_000_000;

    //----------------

    let dims = 768; // can't use everywhere needs to ne manifest?

    let queries: Array1<Simd128> = Array1::from_iter((0..num_queries).map(|_| {
        let embedding = Array1::from_iter((0..dims).map(|_| random::<f32>()));
        to_one_bit_128(embedding)
    }));

    let data: Array1<Simd128> = Array1::from_iter((0..num_data).map(|_| {
        let embedding = Array1::from_iter((0..dims).map(|_| random::<f32>()));
        to_one_bit_128(embedding)
    }));

    let now = Instant::now();

    // Do a brute force of query bitmaps against the data bitmaps

    let distances = generate_hamming_dists(queries, data);

    let after = Instant::now();

    println!("Sum of distances is {:?}", distances.iter().flatten().sum::<u32>());


    println!(
        "Time per Hamming 768 dim query 1_000_000 dists: {} ns",
        ((after - now).as_nanos() as f64) / num_queries as f64
    );

    Ok(())
}

fn generate_hamming_dists(queries: Array1<Simd128>, data: Array1<Simd128>) -> Vec<Vec<u32>> {
    queries
        .iter()
        .map(|query| {
            data.iter()
                .map(|data| one_bit_similarity_128(&query, &data))
                .collect::<Vec<u32>>()
        })
        .collect::<Vec<Vec<u32>>>()
}
