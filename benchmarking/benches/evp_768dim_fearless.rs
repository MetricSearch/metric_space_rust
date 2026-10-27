use bits::container::BitsContainer;
use criterion::{criterion_group, criterion_main};
use criterion::{BenchmarkId, Criterion};
use divan::black_box;
use fearless_simd::{dispatch, prelude::*, u64x8, Level};
use fearless_simd_macros::simd;
use ndarray::Array1;
use rand::random;
use std::ops::BitAnd;

#[derive(Copy, Clone)]
pub struct EvpBits<S: Simd> {
    ones: [u64x8<S>; 2],
    negative_ones: [u64x8<S>; 2],
}

impl<S: Simd> ExtractToken for EvpBits<S> {
    type S = S;

    #[inline]
    fn token(&self) -> S {
        self.ones[0].token()
    }
}

#[simd]
fn new_random<S: Simd>(simd: S) -> EvpBits<S> {
    let evp = bits::evp::EvpBits::<bits::container::Simd256x4, 768>::from_embedding(
        Array1::from_iter((0..768).map(|_| random::<f32>())),
        420,
    );

    let ones_vec = evp
        .ones
        .iter()
        .flat_map(|a| a.into_u64_iter())
        .collect::<Vec<_>>();

    let ones = [
        u64x8::from_slice(simd, &ones_vec[..8]),
        u64x8::from_slice(simd, &ones_vec[8..]),
    ];

    let negative_ones_vec = evp
        .negative_ones
        .iter()
        .flat_map(|a| a.into_u64_iter())
        .collect::<Vec<_>>();

    let negative_ones = [
        u64x8::from_slice(simd, &negative_ones_vec[..8]),
        u64x8::from_slice(simd, &negative_ones_vec[8..]),
    ];

    EvpBits {
        ones,
        negative_ones,
    }
}

#[simd]
fn similarity<S: Simd, const W: u64>(a: EvpBits<S>, b: EvpBits<S>) -> u64 {
    let aa = a.ones[0].bitand(b.ones[0]).count_ones().reduce_sum()
        + a.ones[1].bitand(b.ones[1]).count_ones().reduce_sum();

    let bb = a.negative_ones[0]
        .bitand(b.negative_ones[0])
        .count_ones()
        .reduce_sum()
        + a.negative_ones[1]
            .bitand(b.negative_ones[1])
            .count_ones()
            .reduce_sum();

    let cc = a.ones[0]
        .bitand(b.negative_ones[0])
        .count_ones()
        .reduce_sum()
        + a.ones[1]
            .bitand(b.negative_ones[1])
            .count_ones()
            .reduce_sum();

    let dd = b.ones[0]
        .bitand(a.negative_ones[0])
        .count_ones()
        .reduce_sum()
        + b.ones[1]
            .bitand(a.negative_ones[1])
            .count_ones()
            .reduce_sum();

    // println!( "a {:?} b {:?}", a, b) ;

    // adding X * 256 * 2 means the result must be positive since second term is maximally X * 256 * 2  ( BitVecSimd<[u64x4; X],4> )
    // min is zero (not in practice) max is 2048 (not in practice).
    // println!("A={} B={} C={} D={} result ={}", A, B, C, D, (A + B+X*256*2) - (C + D ));

    (aa + bb + W * 2) - (cc + dd)
}

fn evp_768dim(c: &mut Criterion) {
    let mut group = c.benchmark_group("evp_768dim_fearless");

    group.bench_function(BenchmarkId::from_parameter(768), |bencher| {
        let size = 768;

        let level = Level::new();
        dispatch!(level, simd => {

        let query = new_random(simd);
        let data = new_random(simd);

               bencher.iter(|| {
                  let res = similarity::<_, 768>(black_box(query), black_box(data));
        black_box(res);
               });
        });
    });
}

criterion_group!(benches, evp_768dim);
criterion_main!(benches);
