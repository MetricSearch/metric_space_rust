use std::rc::Rc;
use bitvec_simd::BitVecSimd;
use ndarray::{Array, Array1, Array2, ArrayView, Axis, Ix1, Ix2};
use dao::csv_f32_loader::csv_f32_load;
use dao::{dao_from_dir, Dao32};
use divan::{black_box, counter::BytesCount, AllocProfiler, Bencher};
use bits::{embedding_to_bitrep,hamming_distance};

fn main() {
    divan::main();

}

#[divan::bench]
fn bench(bencher: Bencher) { // bencher: Bencher

    let num_queries = 10_000;
    let num_data = 1_000_000 - num_queries;

    let dao: Rc<Dao32> = Rc::new(dao_from_dir("/Volumes/Data/RUST_META/mf_dino2_csv/meta_data.txt", num_data, num_queries));

    let query = embedding_to_bitrep(dao.get_query(0).view());
    let data = embedding_to_bitrep(dao.get_datum(0).view());

    bencher
        .bench(|| {
            hamming_distance(&query, &data)
        });
}
