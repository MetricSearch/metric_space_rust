use bits::container::Simd128;
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
    let query = EvpBits::<Simd128, 100>::from_embedding(
        Array1::from_iter((0..100).map(|_| random::<f32>())),
        67,
    );
    let data = EvpBits::<Simd128, 100>::from_embedding(
        Array1::from_iter((0..100).map(|_| random::<f32>())),
        67,
    );

    bencher.bench(|| {
        let res = similarity::<Simd128, 100>(black_box(&query), black_box(&data));
        black_box(res);
    });
}
