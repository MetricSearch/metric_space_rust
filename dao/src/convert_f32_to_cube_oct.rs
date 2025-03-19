use std::rc::Rc;
use bitvec_simd::BitVecSimd;
use ndarray::Array1;
use wide::u64x4;
use bits::{f32_embedding_to_cubeoct_bitrep};
use crate::Dao;

pub fn to_cube_oct_dao(f32_dao: Rc<Dao<Array1<f32>>>) -> Rc<Dao<BitVecSimd<[u64x4; 4], 4>>> {
    let bit_embeddings = to_cube_oct_embeddings(&f32_dao.embeddings);

    let mut meta= f32_dao.meta.clone();
    meta.path_to_data = "none".parse().unwrap();
    meta.data_disk_format = "Cubeoct".parse().unwrap();

    Rc::new( Dao{
        meta,
        num_data: f32_dao.num_data.clone(),
        num_queries: f32_dao.num_queries.clone(),
        embeddings: bit_embeddings,
    })
}

pub fn to_cube_oct_embeddings(embeddings: &Array1<Array1<f32>>) -> Array1<BitVecSimd<[u64x4; 4], 4>> {
    embeddings
        .iter()
        .map( |row| { f32_embedding_to_cubeoct_bitrep(row) } )
        .collect()
}