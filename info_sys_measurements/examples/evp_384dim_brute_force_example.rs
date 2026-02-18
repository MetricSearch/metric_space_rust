use anyhow::Result;
//use std::random::random;
use bits::container::Simd256x2;
use bits::{distance, EvpBits};
use ndarray::Array1;
use rand::random;
use std::time::Instant;

fn main() -> Result<()> {
    let num_data = 1_000_000;

    //----------------

    let dims = 384;

    let query_f32 = Array1::from_iter((0..dims).map(|_| random::<f32>()));
    let query: EvpBits<Simd256x2, 384> = EvpBits::<Simd256x2, 384>::from_embedding(query_f32, 200);

    let data: Array1<EvpBits<Simd256x2, 384>> = Array1::from_iter((0..num_data).map(|_| {
        let embedding = Array1::from_iter((0..dims).map(|_| random::<f32>()));
        EvpBits::<Simd256x2, 384>::from_embedding(embedding, 200)
    }));

    let now = Instant::now();

    // Do a brute force of query bitmaps against the data bitmaps

    let bsp_distances = generate_bsp_dists(query, data);

    let after = Instant::now();

    eprintln!("Last distance is {:?}", bsp_distances.iter().last());

    println!("BSP 384:\t{}", ((after - now).as_nanos() as f64));

    Ok(())
}

fn generate_bsp_dists(
    query: EvpBits<Simd256x2, 384>,
    data: Array1<EvpBits<Simd256x2, 384>>,
) -> Vec<usize> {
    data.iter()
        .map(|datum| distance(&query, &datum))
        .collect::<Vec<usize>>()
}
