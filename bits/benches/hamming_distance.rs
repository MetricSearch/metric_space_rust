use bits::{embedding_to_bitrep, hamming_distance};
use bitvec_simd::BitVecSimd;
use dao::csv_f32_loader::{csv_f32_load, dao_from_csv_dir};
use dao::Dao;
use divan::{black_box, counter::BytesCount, AllocProfiler, Bencher};
use ndarray::{Array, Array1, Array2, ArrayView, Axis, Ix1, Ix2};
use std::rc::Rc;

fn main() {
    divan::main();
}

#[divan::bench]
fn bench(bencher: Bencher) {
    // bencher: Bencher

    let num_queries = 10_000;
    let num_data = 1_000_000 - num_queries;

    let dao: Rc<Dao<Array1<f32>>> = Rc::new(dao_from_csv_dir(
        "/Volumes/Data/RUST_META",
        num_data,
        num_queries,
    ).unwrap());

    let query = embedding_to_bitrep(dao.get_query(0).view());
    let data = embedding_to_bitrep(dao.get_datum(0).view());

    bencher.bench(|| {
        let res = hamming_distance(black_box(&query), black_box(&data));
        black_box(res);
    });
}
