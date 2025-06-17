use crate::Dao;
use bitvec_simd::BitVecSimd;
use ndarray::Array1;
use std::rc::Rc;
use wide::u64x4;
use bits::cubic::f32_embedding_to_cubic_bitrep;

pub fn to_cubic_dao(f32_dao: Rc<Dao<Array1<f32>>>) -> Rc<Dao<BitVecSimd<[u64x4; 4], 4>>> {
    let bit_embeddings = to_cubic_embeddings(&f32_dao.embeddings);

    let mut meta = f32_dao.meta.clone();
    meta.path_to_data = "none".parse().unwrap();
    meta.data_disk_format = "Cubic".parse().unwrap();

    Rc::new(Dao {
        meta,
        num_data: f32_dao.num_data.clone(),
        num_queries: f32_dao.num_queries.clone(),
        embeddings: bit_embeddings,
    })
}

pub fn to_cubic_embeddings(embeddings: &Array1<Array1<f32>>) -> Array1<BitVecSimd<[u64x4; 4], 4>> {
    embeddings
        .iter()
        .map(|row| f32_embedding_to_cubic_bitrep(row))
        .collect()
}
