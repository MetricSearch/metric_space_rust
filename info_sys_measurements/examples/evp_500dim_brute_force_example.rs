use anyhow::Result;
//use std::random::random;
use bits::container::Simd256x2;
use bits::{EvpBits, distance};
use ndarray::Array1;
use rand::random;
use rayon::prelude::*;
use std::time::Instant;

fn main() -> Result<()> {
    let num_queries = 100;
    let num_data = 1_000_000;

    //----------------

    let dims = 500; // can't use everywhere needs to ne manifest?

    let queries: Array1<EvpBits<Simd256x2, 500>> = Array1::from_iter((0..num_queries).map(|_| {
        let embedding = Array1::from_iter((0..dims).map(|_| random::<f32>()));
        EvpBits::<Simd256x2, 500>::from_embedding(embedding, 333)
    }));

    let data: Array1<EvpBits<Simd256x2, 500>> = Array1::from_iter((0..num_data).map(|_| {
        let embedding = Array1::from_iter((0..dims).map(|_| random::<f32>()));
        EvpBits::<Simd256x2, 500>::from_embedding(embedding, 333)
    }));

    let now = Instant::now();

    // Do a brute force of query bitmaps against the data bitmaps

    let bsp_distances = generate_bsp_dists(queries, data);

    let after = Instant::now();

    println!("Sum of distances is {:?}", bsp_distances.iter().flatten().sum::<usize>());


    println!(
        "Time per BSP 500 dim query 1_000_000 dists: {} ns",
        ((after - now).as_nanos() as f64) / num_queries as f64
    );

    Ok(())
}

fn generate_bsp_dists(
    queries: Array1<EvpBits<Simd256x2, 500>>,
    data: Array1<EvpBits<Simd256x2, 500>>,
) -> Vec<Vec<usize>> {
    queries
        .iter()
        .map(|query| {
            data.iter()
                .map(|data| distance(&query, &data))
                .collect::<Vec<usize>>()
        })
        .collect::<Vec<Vec<usize>>>()
}
