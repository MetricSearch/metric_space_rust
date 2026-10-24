//use bitvec_simd::BitVecSimd;
use benchmarking::distances::euc_view;
use dao::csv_dao_loader::dao_from_csv_dir;
use dao::Dao;
use divan::{black_box, Bencher};
use ndarray::Array1;
use std::rc::Rc;
use wide::f32x8;

fn main() {
    divan::main();
}

#[divan::bench]
fn bench(bencher: Bencher) {
    // bencher: Bencher

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

    let query = dao.get_query(0);
    let data = dao.get_datum(0);

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

    bencher.bench(|| {
        let res = euc_view(black_box(&query), black_box(&data));
        black_box(res);
    });
}
