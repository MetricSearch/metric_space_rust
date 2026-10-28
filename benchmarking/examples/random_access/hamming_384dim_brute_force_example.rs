use anyhow::Result;
use benchmarking::distances::one_bit_similarity_256 as one_bit_similarity_128;
use benchmarking::util::perm_system_time;
use bits::container::BitsContainer;
use bits::container::Simd128;
use bits::container::Simd256x2;
use ndarray::Array1;
use rand::random;
use std::time::Instant;

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

    let dims = 384; // can't use everywhere needs to ne manifest?

    let query: Simd256x2 = {
        let embedding = Array1::from_iter((0..dims).map(|_| random::<f32>()));
        to_one_bit_128(embedding)
    };

    let data: Array1<Simd256x2> = Array1::from_iter((0..num_data).map(|_| {
        let embedding = Array1::from_iter((0..dims).map(|_| random::<f32>()));
        to_one_bit_128(embedding)
    }));

    let perm = perm_system_time(num_data);

    let now = Instant::now();

    // Do a brute force of query bitmaps against the data bitmaps
    let distances = generate_hamming_dists(&query, &data, &perm);

    let after = Instant::now();

    eprintln!("Last distance is {:?}", distances.iter().last());

    println!(
        "Hamming 384:\t{} ns",
        ((after - now).as_nanos() as f64) as f64
    );

    Ok(())
}

fn generate_hamming_dists(query: &Simd256x2, data: &Array1<Simd256x2>, perm: &[usize]) -> Vec<u32> {
    let mut dists = vec![0; data.len()];

    // Write sequentially, read randomly
    for (seq_id, rand_id) in perm.iter().enumerate() {
        let d = &data[*rand_id];
        dists[seq_id] = one_bit_similarity_128(query, d);
    }

    dists
}
