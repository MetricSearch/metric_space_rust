use bits::{f32_embedding_to_bitrep, hamming_distance};
//use bitvec_simd::BitVecSimd;
use dao::csv_f32_loader::{dao_from_csv_dir};
use dao::Dao;
use divan::{black_box, Bencher};
use ndarray::{Array1};
use std::rc::Rc;

fn main() {
    divan::main();
}

#[divan::bench]
fn bench(bencher: Bencher) {

    let num_queries = 10_000;
    let num_data = 1_000_000 - num_queries;

    let dao: Rc<Dao<Array1<f32>>> =
        Rc::new(dao_from_csv_dir("/Volumes/Data/RUST_META/mf_dino2_csv/", num_data, num_queries).unwrap());

    let query = f32_embedding_to_bitrep(dao.get_query(0));
    let data = f32_embedding_to_bitrep(dao.get_datum(0));

    bencher.bench(|| {
        let res = hamming_distance(black_box(&query), black_box(&data));
        black_box(res);
    });
}
