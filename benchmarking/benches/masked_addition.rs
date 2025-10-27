use bits::evp::{masked_add_selectors, masked_addition_al, masked_addition_gpt};
use bits::{container::Simd256x2, similarity, EvpBits};
use dao::csv_dao_loader::dao_from_csv_dir;
use dao::Dao;
use divan::{black_box, Bencher};
use ndarray::Array1;
use std::rc::Rc;

// "/mnt/bulk/datasets/dino"
const CSV_DIR: &str = "/Volumes/Data/RUST_META/mf_dino2_csv/";
const NUM_QUERIES: usize = 10_000;
const NUM_DATA: usize = 1_000_000 - NUM_QUERIES;

fn main() {
    divan::main();
}

#[divan::bench]
fn bench_masked_add(bencher: Bencher) {
    let dao: Rc<Dao<Array1<f32>>> =
        Rc::new(dao_from_csv_dir(CSV_DIR, NUM_DATA, NUM_QUERIES).unwrap());

    let query = dao.get_query(0).view();
    let data = dao.get_datum(0).view();
    let evp_data = EvpBits::<Simd256x2, 384>::from_embedding(data, 200);

    bencher.bench(|| {
        let res = masked_add_selectors::<Simd256x2, 384>(black_box(query), black_box(&evp_data));
        black_box(res);
    });
}

#[divan::bench]
fn bench_masked_addition_gpt(bencher: Bencher) {
    let dao: Rc<Dao<Array1<f32>>> =
        Rc::new(dao_from_csv_dir(CSV_DIR, NUM_DATA, NUM_QUERIES).unwrap());

    let query = dao.get_query(0).view();
    let data = dao.get_datum(0).view();
    let evp_data = EvpBits::<Simd256x2, 384>::from_embedding(data, 200);

    bencher.bench(|| {
        let res = masked_addition_gpt::<Simd256x2, 384>(black_box(query), black_box(&evp_data));
        black_box(res);
    });
}

#[divan::bench]
fn bench_evp_similarity(bencher: Bencher) {
    let dao: Rc<Dao<Array1<f32>>> =
        Rc::new(dao_from_csv_dir(CSV_DIR, NUM_DATA, NUM_QUERIES).unwrap());

    let query = dao.get_query(0).view();
    let evp_query = EvpBits::<Simd256x2, 384>::from_embedding(query, 200);
    let data = dao.get_datum(0).view();
    let evp_data = EvpBits::<Simd256x2, 384>::from_embedding(data, 200);

    bencher.bench(|| {
        let res = similarity::<Simd256x2, 384>(black_box(&evp_query), black_box(&evp_data));
        black_box(res);
    });
}

#[divan::bench]
fn bench_masked_al(bencher: Bencher) {
    let dao: Rc<Dao<Array1<f32>>> =
        Rc::new(dao_from_csv_dir(CSV_DIR, NUM_DATA, NUM_QUERIES).unwrap());

    let query = dao.get_query(0).view();
    let data = dao.get_datum(0).view();
    let evp_data = EvpBits::<Simd256x2, 384>::from_embedding(data, 200);

    bencher.bench(|| {
        let res = masked_addition_al::<Simd256x2, 384>(black_box(query), black_box(&evp_data));
        black_box(res);
    });
}
