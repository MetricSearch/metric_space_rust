use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use dao::csv_dao_loader::dao_from_csv_dir;
use metrics::euc;
use ndarray::s;
use std::hint::black_box;
use std::rc::Rc;

fn bench_pca(c: &mut Criterion) {
    let num_queries = 10_000;
    let num_data = 1_000_000 - num_queries;
    let dims = 384;

    let dao = Rc::new(
        dao_from_csv_dir(
            "/Volumes/Data/RUST_META/mf_dino2_csv/",
            num_data,
            num_queries,
        )
        .unwrap(),
    );

    let mut group = c.benchmark_group("bench_pca");

    for size in [
        dims / 2,  // 192 for 384
        dims / 4,  // 96 fpr 384
        dims / 8,  // 48 for 384
        dims / 16, // 24 for 384
        dims / 32, // 12 for 384
        dims / 64, // 6 for 384
        64,
        32,
    ] {
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &size,
            |bencher, &size| {
                let query = dao.get_query(0).slice(s![0..size]).to_owned();
                let data = dao.get_datum(1).slice(s![0..size]).to_owned();

                bencher.iter(|| {
                    let res = euc(black_box(&query), black_box(&data));
                    black_box(res);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_pca);
criterion_main!(benches);
