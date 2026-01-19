//use bitvec_simd::BitVecSimd;
use dao::csv_dao_loader::dao_from_csv_dir;
use dao::Dao;
use divan::{black_box, Bencher};
use ndarray::Array1;
use std::rc::Rc;

pub fn euc_16bit(a: &Array1<i16>, b: &Array1<i16>) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(a, b)| (a - b).pow(2) as f32)
        .sum()
}

pub fn get_max(data: &Array1<f32>) -> f32 {
    return data.iter().cloned().reduce(f32::max).unwrap();
}

pub fn to_i16_array(array: &Array1<f32>, max_f32: f32) -> Array1<i16> {
    array.mapv(|x| {
        let value = x / max_f32;

        if value.is_nan() {
            // this will never happen
            0
        } else {
            (value * i16::MAX as f32)
                .round()
                .clamp(i16::MIN as f32, i16::MAX as f32) as i16
        }
    })
}

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

    let data_f32 = dao.get_datum(0);
    let max_f32 = get_max(data_f32);
    let query_f32 = dao.get_query(0);

    let query: Array1<i16> = to_i16_array(query_f32, max_f32);
    let data = to_i16_array(data_f32, max_f32);

    bencher.bench(|| {
        let res = euc_16bit(black_box(&query), black_box(&data));
        black_box(res);
    });
}
