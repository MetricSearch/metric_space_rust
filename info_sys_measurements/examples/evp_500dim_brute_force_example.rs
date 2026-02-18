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

    let dims = 500;

    let query_f32 = Array1::from_iter((0..dims).map(|_| random::<f32>()));
    let query: EvpBits<Simd256x2, 500> = EvpBits::<Simd256x2, 500>::from_embedding(query_f32, 333);

    let data: Array1<EvpBits<Simd256x2, 500>> = Array1::from_iter((0..num_data).map(|_| {
        let embedding = Array1::from_iter((0..dims).map(|_| random::<f32>()));
        EvpBits::<Simd256x2, 500>::from_embedding(embedding, 333)
    }));

    let now = Instant::now();

    // Do a brute force of query bitmaps against the data bitmaps

    let bsp_distances = generate_bsp_dists(query, data);

    let after = Instant::now();

    eprintln!("Last distance is {:?}", bsp_distances.iter().last());

    println!("BSP 500:\t{}", ((after - now).as_nanos() as f64));

    Ok(())
}

fn generate_bsp_dists(
    query: EvpBits<Simd256x2, 500>,
    data: Array1<EvpBits<Simd256x2, 500>>,
) -> Vec<usize> {
    data.iter()
        .map(|datum| distance(&query, &datum))
        .collect::<Vec<usize>>()
}
