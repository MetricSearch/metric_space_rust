use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::Array1;
use rand::random;
use std::hint::black_box;

fn bench_scalar_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("bench_scalar_product");

    fn scalar_product(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
        a.iter().zip(b.iter()).map(|(a, b)| (a * b)).sum()
    }

    for size in [100, 384, 500, 768] {
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &size,
            |bencher, &size| {
                let query = Array1::from_iter((0..size).map(|_| random::<f32>()));
                let data = Array1::from_iter((0..size).map(|_| random::<f32>()));

                bencher.iter(|| {
                    let res = scalar_product(black_box(&query), black_box(&data));
                    black_box(res);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_scalar_product);
criterion_main!(benches);
