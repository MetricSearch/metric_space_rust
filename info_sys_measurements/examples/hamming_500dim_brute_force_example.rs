use anyhow::Result;
use bits::container::BitsContainer;
use bits::container::Simd256x2;
use ndarray::Array1;
use rand::random;
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
    let num_data = 1_000_000;

    //----------------

    let dims = 500; // can't use everywhere needs to ne manifest?

    let query: Simd256x2 = {
        let embedding = Array1::from_iter((0..dims).map(|_| random::<f32>()));
        to_one_bit_128(embedding)
    };

    let data: Array1<Simd256x2> = Array1::from_iter((0..num_data).map(|_| {
        let embedding = Array1::from_iter((0..dims).map(|_| random::<f32>()));
        to_one_bit_128(embedding)
    }));

    let now = Instant::now();

    // Do a brute force of query bitmaps against the data bitmaps

    let distances = generate_hamming_dists(query, data);

    let after = Instant::now();

    eprintln!("Last distance is {:?}", distances.iter().last());

    println!(
        "Hamming 500:\t{} ns",
        ((after - now).as_nanos() as f64) as f64
    );

    Ok(())
}

fn generate_hamming_dists(query: Simd256x2, data: Array1<Simd256x2>) -> Vec<u32> {
    data.iter()
        .map(|d| one_bit_similarity_128(&query, d))
        .collect::<Vec<u32>>()
}
