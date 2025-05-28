//use bitvec_simd::BitVecSimd;
use dao::csv_dao_loader::dao_from_csv_dir;
use dao::Dao;
use divan::{black_box, Bencher};
use metrics::euc;
use ndarray::Array1;
use std::rc::Rc;
use utils::index::Index;

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

    let query = dao.get_query(Index::new(0));
    let data = dao.get_datum(Index::new(0));

    bencher.bench(|| {
        for _ in 0..1_000_000 {
            let res = euc(black_box(query), black_box(data));
            black_box(res);
        }
    });
}
