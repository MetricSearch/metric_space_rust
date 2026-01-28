use bits::container::BitsContainer;
use bits::{container::Simd256x2, similarity, EvpBits};
use dao::csv_dao_loader::dao_from_csv_dir;
use dao::Dao;
use divan::{black_box, Bencher};
use ndarray::{Array1, ArrayBase, ArrayView1, ArrayView2, Axis, Data, Ix1};
use std::fmt;
use std::rc::Rc;

fn main() {
    divan::main();
}

fn to_one_bit(embedding: ArrayView1<f32>) -> Simd256x2 {
    let mut bits = Simd256x2::new();
    (0..embedding.len()).for_each(|index| {
        if embedding[index] > 0.0 {
            bits.set_bit(index, true);
        }
    });
    bits
}

fn one_bit_similarity(a: &Simd256x2, b: &Simd256x2) -> u32 {
    384 - a
        .into_u64_iter()
        .zip(b.into_u64_iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum::<u32>()
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

    let query = to_one_bit(dao.get_query(0).view());
    let data = to_one_bit(dao.get_datum(0).view());

    bencher.bench(|| {
        let res = one_bit_similarity(black_box(&query), black_box(&data));
        black_box(res);
    });
}
