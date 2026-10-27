use bits::container::Simd256x4;
use bits::{similarity, EvpBits};
use criterion::{criterion_group, criterion_main};
use criterion::{BenchmarkId, Criterion};
use divan::black_box;
use ndarray::Array1;
use rand::random;

fn evp_768dim(c: &mut Criterion) {
    let mut group = c.benchmark_group("evp_768dim");

    let query = EvpBits::<Simd256x4, 768>::from_embedding(
        Array1::from_iter((0..768).map(|_| random::<f32>())),
        420,
    );
    let data = EvpBits::<Simd256x4, 768>::from_embedding(
        Array1::from_iter((0..768).map(|_| random::<f32>())),
        420,
    );

    group.bench_function(BenchmarkId::from_parameter(768), |bencher| {
        bencher.iter(|| {
            let res = similarity::<Simd256x4, 768>(black_box(&query), black_box(&data));
            black_box(res);
        });
    });
}

criterion_group!(benches, evp_768dim);
criterion_main!(benches);
