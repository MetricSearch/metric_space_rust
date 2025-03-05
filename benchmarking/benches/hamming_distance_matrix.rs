//use bits::{f32_embedding_to_bitrep, hamming_distance};
//use bitvec_simd::BitVecSimd;
use dao::csv_f32_loader::{dao_from_csv_dir};
use dao::Dao;
use divan::{black_box,Bencher};
use ndarray::{Array1, Array2, ArrayView, Axis, Ix1};
use std::rc::Rc;

// This tests speed of f32 based matrix multiplications.

fn main() {
    divan::main();
}

#[divan::bench]
fn bench(bencher: Bencher) {

    let num_queries = 10_000;
    let num_data = 1_000_000 - num_queries;

    let dao: Rc<Dao<Array1<f32>>> =
        Rc::new(dao_from_csv_dir("/Volumes/Data/RUST_META/mf_dino2_csv/", num_data, num_queries).unwrap());

    let queries = dao.get_queries();
    let data = dao.get_data();

    let queries_100 = queries.split_at(Axis(0), 100).0;
    let data_10k = data.split_at(Axis(0), 10000).0;

    let queries_100 = to_2d_array(queries_100,384);
    let data_10k = to_2d_array(data_10k,384);

    bencher.bench(|| {
        let dists = queries_100.dot(&data_10k.t());
        // println!( "size = {:?}", dists.shape() );
        black_box(dists);
    });
}

fn to_2d_array(input : ArrayView<Array1<f32>, Ix1>, n_cols: usize ) -> Array2<f32> {
    let mut data = Vec::new();
    let mut nrows = 0;
    input
        .iter()
        .for_each( |row| { data.extend_from_slice(row.as_slice().unwrap()); nrows += 1; } );
    let arr = Array2::from_shape_vec((nrows, n_cols), data).unwrap();
    arr
}
