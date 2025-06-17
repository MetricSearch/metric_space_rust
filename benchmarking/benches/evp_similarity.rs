use bits::{container::Simd256x2, similarity, EvpBits};
use dao::csv_dao_loader::dao_from_csv_dir;
use dao::Dao;
use divan::{black_box, Bencher};
use ndarray::Array1;
use std::rc::Rc;

fn main() {
    divan::main();
}

#[divan::bench]
fn bench(bencher: Bencher) {
    let num_queries = 10_000;
    let num_data = 1_000_000 - num_queries;

    let dao: Rc<Dao<Array1<f32>>> = Rc::new(
        dao_from_csv_dir(
            "/Volumes/Data/RUST_META/mf_dino2_csv/",
            num_data,
            num_queries,
        )
        .unwrap(),
    );

    let query = EvpBits::<Simd256x2, 384>::from_embedding(dao.get_query(0).view(), 200);
    let data = EvpBits::<Simd256x2, 384>::from_embedding(dao.get_datum(0).view(), 200);

    bencher.bench(|| {
        for _ in 0..1_000_000 {
            let res = similarity::<Simd256x2, 384>(black_box(&query), black_box(&data));
            black_box(res);
        }
    });
}
