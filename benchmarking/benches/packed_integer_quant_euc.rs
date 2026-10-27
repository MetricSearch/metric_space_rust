use benchmarking::distances::{
    euc_16bit, euc_4bit, euc_8bit, to_packed_4bit_vectors, to_u16_array, to_u16x16_vectors,
    to_u4_array, to_u8_array, to_u8x32_vectors,
};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::Array1;
use rand::random;
use std::hint::black_box;

pub fn euc_32bit(a: &Array1<i32>, b: &Array1<i32>) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(a, b)| (a - b).pow(2) as f32)
        .sum()
}

pub fn get_max(data: &Array1<f32>) -> f32 {
    return data.iter().cloned().reduce(f32::max).unwrap();
}

pub fn to_i32_array(array: &Array1<f32>, max_f32: f32) -> Array1<i32> {
    array.mapv(|x| {
        let value = x / max_f32;

        if value.is_nan() {
            // this will never happen
            0
        } else {
            (value * i32::MAX as f32)
                .round()
                .clamp(i32::MIN as f32, i32::MAX as f32) as i32
        }
    })
}

fn bench_multiple_eucs(c: &mut Criterion) {
    let mut group = c.benchmark_group("bench_multiple_bit_eucs");

    for size in [100, 384, 500, 768] {
        for bits in [4, 8, 16, 32] {
            let query_f32 = Array1::from_iter((0..size).map(|_| random::<f32>()));
            let data_f32 = Array1::from_iter((0..size).map(|_| random::<f32>()));
            let max_f32 = get_max(&data_f32);

            let s3 = format!("{}/{}", size, bits);

            group.bench_with_input(BenchmarkId::from_parameter(s3), &size, |bencher, &size| {
                if bits == 4 {
                    let query = to_packed_4bit_vectors(&to_u4_array(&query_f32, max_f32), size);
                    let data = to_packed_4bit_vectors(&to_u4_array(&data_f32, max_f32), size);

                    bencher.iter(|| {
                        let res = euc_4bit(black_box(&query), black_box(&data));
                        black_box(res);
                    });
                }

                if bits == 8 {
                    let query = to_u8x32_vectors(&to_u8_array(&query_f32, max_f32), size);
                    let data = to_u8x32_vectors(&to_u8_array(&data_f32, max_f32), size);

                    bencher.iter(|| {
                        let res = euc_8bit(black_box(&query), black_box(&data));
                        black_box(res);
                    });
                }

                if bits == 16 {
                    let query = to_u16x16_vectors(&to_u16_array(&query_f32, max_f32), size);
                    let data = to_u16x16_vectors(&to_u16_array(&data_f32, max_f32), size);

                    bencher.iter(|| {
                        let res = euc_16bit(black_box(&query), black_box(&data));
                        black_box(res);
                    });
                }

                if bits == 32 {
                    let query: Array1<i32> = to_i32_array(&query_f32, max_f32);
                    let data = to_i32_array(&data_f32, max_f32);

                    bencher.iter(|| {
                        let res = euc_32bit(black_box(&query), black_box(&data));
                        black_box(res);
                    });
                }
            });
        }
    }

    group.finish();
}

criterion_group!(benches, bench_multiple_eucs);
criterion_main!(benches);
