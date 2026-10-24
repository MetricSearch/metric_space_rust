use benchmarking::distances::euc_view;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::Array1;
use rand::random;
use std::hint::black_box;
use wide::f32x8;

fn bench_multiple_eucs(c: &mut Criterion) {
    let mut group = c.benchmark_group("bench_multiple_eucs");

    for size in [100, 384, 500, 768] {
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &size,
            |bencher, &size| {
                let query = Array1::from_iter((0..size).map(|_| random::<f32>()));
                let data = Array1::from_iter((0..size).map(|_| random::<f32>()));

                let query: Vec<f32x8> = query
                    .as_slice()
                    .unwrap()
                    .chunks(8)
                    .map(f32x8::from)
                    .collect();
                let data: Vec<f32x8> = data
                    .as_slice()
                    .unwrap()
                    .chunks(8)
                    .map(f32x8::from)
                    .collect();

                bencher.iter(|| {
                    let res = euc_view(black_box(&query), black_box(&data));
                    black_box(res);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_multiple_eucs);
criterion_main!(benches);
