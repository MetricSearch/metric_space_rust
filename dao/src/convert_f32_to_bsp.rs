use crate::{Dao, DaoMatrix};
use bits::{container::BitsContainer, f32_embeddings_to_bsp, EvpBits};
use std::rc::Rc;

pub fn f32_dao_to_bsp<C: BitsContainer, const W: usize>(
    f32_dao: Rc<DaoMatrix<f32>>,
    non_zeros: usize,
) -> Rc<Dao<EvpBits<C, W>>> {
    let bit_embeddings = f32_embeddings_to_bsp(&f32_dao.embeddings, non_zeros);

    let mut meta = f32_dao.meta.clone();
    meta.path_to_data = "none".parse().unwrap();
    meta.data_disk_format = "EVP()".parse().unwrap();

    Rc::new(Dao {
        meta,
        num_data: f32_dao.num_data.clone(),
        num_queries: f32_dao.num_queries.clone(),
        embeddings: bit_embeddings,
    })
}
