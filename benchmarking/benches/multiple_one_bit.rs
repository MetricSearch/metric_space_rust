use bits::container::Simd256x2;
use bits::container::{BitsContainer, Simd128, Simd256x4};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use divan::black_box;
use ndarray::{Array1, ArrayView1};
use rand::random;

fn to_one_bit_128(embedding: ArrayView1<f32>) -> Simd128 {
    let mut bits = Simd128::default();
    (0..embedding.len()).for_each(|index| {
        if embedding[index] > 0.0 {
            bits.set_bit(index, true);
        }
    });
    bits
}

fn to_one_bit_256(embedding: ArrayView1<f32>) -> Simd256x2 {
    let mut bits = Simd256x2::new();
    (0..embedding.len()).for_each(|index| {
        if embedding[index] > 0.0 {
            bits.set_bit(index, true);
        }
    });
    bits
}

fn to_one_bit_786(embedding: ArrayView1<f32>) -> Simd256x4 {
    let mut bits = Simd256x4::new();
    (0..embedding.len()).for_each(|index| {
        if embedding[index] > 0.0 {
            bits.set_bit(index, true);
        }
    });
    bits
}

fn one_bit_similarity_128(a: &Simd128, b: &Simd128) -> u32 {
    a.into_u64_iter()
        .zip(b.into_u64_iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum::<u32>()
}

fn one_bit_similarity_256(a: &Simd256x2, b: &Simd256x2) -> u32 {
    a.into_u64_iter()
        .zip(b.into_u64_iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum::<u32>()
}

fn one_bit_similarity_768(a: &Simd256x4, b: &Simd256x4) -> u32 {
    a.into_u64_iter()
        .zip(b.into_u64_iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum::<u32>()
}

fn bench_one_bit(c: &mut Criterion) {
    let mut group = c.benchmark_group("bench_multiple_one_bit");

    for size in [100, 384, 500, 768] {
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &size,
            |bencher, &size| {
                let query = Array1::from_iter((0..size).map(|_| random::<f32>()));
                let data = Array1::from_iter((0..size).map(|_| random::<f32>()));

                if size == 100 {
                    let query = to_one_bit_128(query.view());
                    let data = to_one_bit_128(data.view());

                    bencher.iter(|| {
                        let res = one_bit_similarity_128(black_box(&query), black_box(&data));
                        black_box(res);
                    });
                }

                if size == 384 || size == 500 {
                    let query = to_one_bit_256(query.view());
                    let data = to_one_bit_256(data.view());

                    bencher.iter(|| {
                        let res = one_bit_similarity_256(black_box(&query), black_box(&data));
                        black_box(res);
                    });
                }

                if size == 768 {
                    let query = to_one_bit_786(query.view());
                    let data = to_one_bit_786(data.view());

                    bencher.iter(|| {
                        let res = one_bit_similarity_768(black_box(&query), black_box(&data));
                        black_box(res);
                    });
                }
            },
        );
    }
}

criterion_group!(benches, bench_one_bit);
criterion_main!(benches);
