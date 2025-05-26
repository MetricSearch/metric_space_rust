use bits::{f32_embedding_to_evp, whamming_distance};
//use bitvec_simd::BitVecSimd;
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

    let query = f32_embedding_to_evp::<3>(dao.get_query(0), 200);
    let data = f32_embedding_to_evp::<3>(dao.get_datum(0), 200);

    bencher.bench(|| {
        for _ in 0..1_000_000 {
            let res = whamming_distance::<3>(black_box(&query), black_box(&data));
            black_box(res);
        }
    });
}
