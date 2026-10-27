use anyhow::Result;
use benchmarking::distances::one_bit_similarity_768 as one_bit_similarity_128;
use bits::container::BitsContainer;
use bits::container::Simd256x4;
use ndarray::Array1;
use rand::random;
use std::time::Instant;

fn to_one_bit_128(embedding: Array1<f32>) -> Simd256x4 {
    let mut bits = Simd256x4::default();
    (0..embedding.len()).for_each(|index| {
        if embedding[index] > 0.5 {
            bits.set_bit(index, true);
        }
    });
    bits
}

fn main() -> Result<()> {
    let num_data = 1_000_000;

    //----------------

    let dims = 768; // can't use everywhere needs to ne manifest?

    let query: Simd256x4 = {
        let embedding = Array1::from_iter((0..dims).map(|_| random::<f32>()));
        to_one_bit_128(embedding)
    };

    let data: Array1<Simd256x4> = Array1::from_iter((0..num_data).map(|_| {
        let embedding = Array1::from_iter((0..dims).map(|_| random::<f32>()));
        to_one_bit_128(embedding)
    }));

    let now = Instant::now();

    // Do a brute force of query bitmaps against the data bitmaps

    let distances = generate_hamming_dists(&query, &data);

    let after = Instant::now();

    eprintln!("Last distance is {:?}", distances.iter().last());

    println!(
        "Hamming 768:\t{} ns",
        ((after - now).as_nanos() as f64) as f64
    );

    Ok(())
}

fn generate_hamming_dists(query: &Simd256x4, data: &Array1<Simd256x4>) -> Vec<u32> {
    data.iter()
        .map(|d| one_bit_similarity_128(query, d))
        .collect::<Vec<u32>>()
}
