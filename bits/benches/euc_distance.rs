use std::rc::Rc;
use bitvec_simd::BitVecSimd;
use ndarray::{Array, Array1, Array2, ArrayView, Axis, Ix1, Ix2};
use dao::csv_f32_loader::csv_f32_load;
use dao::{Dao32};
use divan::{black_box, counter::BytesCount, AllocProfiler, Bencher};
use bits::{embedding_to_bitrep,hamming_distance};
use metrics::euc;

fn main() {
    divan::main();

}

#[divan::bench]
fn bench(bencher: Bencher) { // bencher: Bencher

    let num_queries = 10_000;
    let num_data = 1_000_000 - num_queries;

    let dao: Rc<Dao32> = Rc::new(Dao32::dao_from_csv_dir("/Volumes/Data/RUST_META/mf_dino2_csv/", num_data, num_queries).unwrap());

    let query = dao.get_query(0);
    let query = query.view();
    let data = dao.get_datum(0);
    let data = data.view();

    bencher
        .bench(|| {
            let res = euc(black_box(query), black_box(data));
            black_box(res);
        });
}
