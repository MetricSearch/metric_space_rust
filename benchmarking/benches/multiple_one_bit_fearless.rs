use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use divan::black_box;
use fearless_simd::{dispatch, prelude::*, u64x8, Level};
use fearless_simd_macros::simd;
use ndarray::{Array1, ArrayView1};
use rand::random;
use std::ops::BitXor;

#[simd]
fn to_one_bit<S: Simd>(simd: S, embedding: ArrayView1<f32>) -> [u64x8<S>; 2] {
    use bits::container::BitsContainer;

    let mut bits = bits::container::Simd256x4::new();
    (0..embedding.len()).for_each(|index| {
        if embedding[index] > 0.0 {
            bits.set_bit(index, true);
        }
    });

    [
        u64x8::from_slice(simd, &bits.into_u64_iter().collect::<Vec<_>>()[..8]),
        u64x8::from_slice(simd, &bits.into_u64_iter().collect::<Vec<_>>()[8..]),
    ]
}

#[simd]
fn one_bit_similarity_768<S: Simd>(simd: S, a: [u64x8<S>; 2], b: [u64x8<S>; 2]) -> u32 {
    // a.into_u64_iter()
    //     .zip(b.into_u64_iter())
    //     .map(|(x, y)| (x ^ y).count_ones())
    //     .sum::<u32>()

    a[0].bitxor(b[0]).count_ones().reduce_sum() as u32
        + a[1].bitxor(b[1]).count_ones().reduce_sum() as u32
}

fn bench_one_bit(c: &mut Criterion) {
    let mut group = c.benchmark_group("bench_multiple_one_bit_fearless");

    group.bench_function(BenchmarkId::from_parameter(768), |bencher| {
        let size = 768;

        let query = Array1::from_iter((0..size).map(|_| random::<f32>()));
        let data = Array1::from_iter((0..size).map(|_| random::<f32>()));

        let level = Level::new();
        dispatch!(level, simd => {
        let query = to_one_bit(simd, query.view());
               let data = to_one_bit(simd, data.view());

               bencher.iter(|| {
                   let res = one_bit_similarity_768(simd, black_box(query), black_box(data));
                   black_box(res);
               });
        });
    });
}

criterion_group!(benches, bench_one_bit);
criterion_main!(benches);
