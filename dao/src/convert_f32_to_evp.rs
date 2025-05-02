use std::rc::Rc;
use bitvec_simd::BitVecSimd;
use ndarray::Array1;
use wide::u64x4;
use bits::{f32_embedding_to_evp};
use crate::Dao;

pub fn f32_dao_to_evp<const D: usize>(f32_dao: Rc<Dao<Array1<f32>>>, non_zeros: usize) -> Rc<Dao<BitVecSimd<[u64x4; D], 4>>> {
    let bit_embeddings = f32_embeddings_to_evp(&f32_dao.embeddings, non_zeros);

    let mut meta= f32_dao.meta.clone();
    meta.path_to_data = "none".parse().unwrap();
    meta.data_disk_format = "EVP()".parse().unwrap();

    Rc::new( Dao{
        meta,
        num_data: f32_dao.num_data.clone(),
        num_queries: f32_dao.num_queries.clone(),
        embeddings: bit_embeddings,
    })
}

pub fn f32_embeddings_to_evp<const D: usize>(embeddings: &Array1<Array1<f32>>, non_zeros: usize) -> Array1<BitVecSimd<[u64x4; D], 4>> {
    embeddings
        .iter()
        .map( |row| { f32_embedding_to_evp::<D>(row,non_zeros) } )
        .collect()
}

fn _example(f32_dao: Rc<Dao<Array1<f32>>>, non_zeros: usize) {

    let _cube_dao = f32_dao_to_evp::<4>(f32_dao.clone(), non_zeros);

}

