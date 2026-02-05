use bits::{container::Simd256x2, similarity, EvpBits};
use dao::Dao;
use divan::{black_box, Bencher};
use ndarray::Array1;
use rand::random;
use std::rc::Rc;

fn main() {
    divan::main();
}
#[divan::bench]
fn bench(bencher: Bencher) {
    let query = EvpBits::<Simd256x2, 500>::from_embedding(
        Array1::from_iter((0..500).map(|_| random::<f32>())),
        333,
    );
    let data = EvpBits::<Simd256x2, 500>::from_embedding(
        Array1::from_iter((0..500).map(|_| random::<f32>())),
        333,
    );

    bencher.bench(|| {
        let res = similarity::<Simd256x2, 500>(black_box(&query), black_box(&data));
        black_box(res);
    });
}
