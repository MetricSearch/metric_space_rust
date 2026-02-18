use anyhow::Result;
use bits::container::BitsContainer;
use bits::container::Simd128;
use bits::container::Simd256x2;
use ndarray::Array1;
use rand::random;
use rayon::prelude::*;
use std::time::Instant;

fn one_bit_similarity_128(a: &Simd256x2, b: &Simd256x2) -> u32 {
    // a.into_u64_iter()
    //     .zip(b.into_u64_iter())
    //     .map(|(x, y)| (x ^ y).count_ones())
    //     .sum::<u32>()

    a.xor(b).count_ones() as u32
}

fn to_one_bit_128(embedding: Array1<f32>) -> Simd256x2 {
    let mut bits = Simd256x2::default();
    (0..embedding.len()).for_each(|index| {
        if embedding[index] > 0.5 {
            bits.set_bit(index, true);
        }
    });
    bits
}

fn main() -> Result<()> {
    let num_queries = 100;
    let num_data = 1_000_000;

    //----------------

    let dims = 384; // can't use everywhere needs to ne manifest?

    let queries: Array1<Simd256x2> = Array1::from_iter((0..num_queries).map(|_| {
        let embedding = Array1::from_iter((0..dims).map(|_| random::<f32>()));
        to_one_bit_128(embedding)
    }));

    let data: Array1<Simd256x2> = Array1::from_iter((0..num_data).map(|_| {
        let embedding = Array1::from_iter((0..dims).map(|_| random::<f32>()));
        to_one_bit_128(embedding)
    }));

    let now = Instant::now();

    // Do a brute force of query bitmaps against the data bitmaps
    let distances = generate_hamming_dists(queries, data);

    let after = Instant::now();

    println!("Last distance is {:?}", distances.iter().flatten().last());

    println!(
        "Time per Hamming 384 dim query 1_000_000 dists: {} ns",
        ((after - now).as_nanos() as f64) / num_queries as f64
    );

    Ok(())
}

fn generate_hamming_dists(queries: Array1<Simd256x2>, data: Array1<Simd256x2>) -> Vec<Vec<u32>> {
    queries
        .par_iter()
        .map(|query| {
            data.iter()
                .map(|data| one_bit_similarity_128(&query, &data))
                .collect::<Vec<u32>>()
        })
        .collect::<Vec<Vec<u32>>>()
}
