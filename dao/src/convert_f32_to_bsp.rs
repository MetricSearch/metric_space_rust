use std::rc::Rc;
use ndarray::Array1;
use bits::{f32_embeddings_to_bsp, Bsp};
use crate::{Dao, DaoMatrix};

pub fn f32_dao_to_bsp<const D: usize>(f32_dao: Rc<DaoMatrix<f32>>, non_zeros: usize) -> Rc<Dao<
Bsp<D>>> {
    let bit_embeddings = f32_embeddings_to_bsp::<D>(&f32_dao.embeddings, non_zeros);

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


