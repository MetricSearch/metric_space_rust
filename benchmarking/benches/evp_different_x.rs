use bits::{container::Simd256x2, similarity, EvpBits};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use dao::csv_dao_loader::dao_from_csv_dir;
use std::hint::black_box;

use std::rc::Rc;

fn bench_different_x(c: &mut Criterion) {
    let num_queries = 10_000;
    let num_data = 1_000_000 - num_queries;

    let dao = Rc::new(
        dao_from_csv_dir(
            "/Volumes/Data/RUST_META/mf_dino2_csv/",
            num_data,
            num_queries,
        )
        .unwrap(),
    );

    let mut group = c.benchmark_group("bench_different_x");

    for bits in (10..380).step_by(10) {
        group.bench_with_input(
            BenchmarkId::from_parameter(bits),
            &bits,
            |bencher, &bits| {
                let query =
                    EvpBits::<Simd256x2, 384>::from_embedding(dao.get_query(0).view(), bits);
                let data = EvpBits::<Simd256x2, 384>::from_embedding(dao.get_datum(0).view(), bits);

                bencher.iter(|| {
                    let res = similarity::<Simd256x2, 384>(black_box(&query), black_box(&data));
                    black_box(res);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_different_x);
criterion_main!(benches);
