use std::rc::Rc;
use bitvec_simd::BitVecSimd;
use ndarray::{Array, Array1, Array2, ArrayView, Axis, Ix1, Ix2};
use dao::csv_loader::csv_loader;
use dao::Dao;
use divan::{black_box, counter::BytesCount, AllocProfiler, Bencher};

fn main() {
    divan::main();

}

#[divan::bench]
fn bench(bencher: Bencher) { // bencher: Bencher

    let num_queries = 10_000;
    let num_data = 1_000_000 - num_queries;

    let dao: Rc<Dao> = Rc::new(Dao::new("/Volumes/data/mf_dino2_csv/mf_dino2.csv", num_data, num_queries, &csv_loader).unwrap());

    let query = embedding_to_bitrep(dao.query(0).view());
    let data = embedding_to_bitrep(dao.get(0).view());

    bencher
        .bench(|| {
            hamming_distance(&query, &data)
        });
}

// TODO below here copied from try_out - refactor later
fn embedding_to_bitrep(embedding: ArrayView<f32, Ix1>) -> BitVecSimd<[wide::u64x4; 4], 4> {
    BitVecSimd::from_bool_iterator(embedding.iter().map(|&x| x < 0.0 ) )
}

fn hamming_distance(a: &BitVecSimd<[wide::u64x4; 4], 4>, b: &BitVecSimd<[wide::u64x4; 4], 4> ) -> usize {
    //assert_eq!(a.len(), b.len());
    a.xor_cloned(&b).count_ones()
}
