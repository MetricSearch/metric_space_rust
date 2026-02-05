use bits::container::Simd256x4;
use bits::{container::Simd256x2, similarity, EvpBits};
use dao::Dao;
use divan::{black_box, Bencher};
use ndarray::Array1;
use rand::random;

fn main() {
    divan::main();
}

#[divan::bench]
fn bench(bencher: Bencher) {
    let query = EvpBits::<Simd256x4, 768>::from_embedding(
        Array1::from_iter((0..768).map(|_| random::<f32>())),
        420,
    );
    let data = EvpBits::<Simd256x4, 768>::from_embedding(
        Array1::from_iter((0..768).map(|_| random::<f32>())),
        420,
    );

    bencher.bench(|| {
        let res = similarity::<Simd256x4, 768>(black_box(&query), black_box(&data));
        black_box(res);
    });
}
