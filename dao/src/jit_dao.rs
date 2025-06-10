use hdf5::Dataset;
use hdf5::File;
use ndarray::s;
use ndarray::Array1;
use ndarray::Ix;
use std::marker::PhantomData;

use crate::DaoMetaData;
use crate::Normed;

pub struct JitDao<T> {
    _marker: PhantomData<T>,
    meta: DaoMetaData,
    data: Dataset,
    queries: Dataset,
    num_data: Ix,
    num_queries: usize,
}

impl JitDao<f32> {
    pub fn load(
        data_path: &str,
        num_records_required: usize, // zero if all the data
        num_queries: usize,
    ) -> anyhow::Result<Self> {
        let file = File::open(data_path)?; // open for reading
        let data = file.dataset("train")?; // the data
        let queries = file.group("otest")?.dataset("queries")?;

        let train_size = data.shape()[0];

        if num_records_required > train_size {
            log::error!("Too many records requested")
        }
        let num_records = if num_records_required == 0 {
            train_size
        } else {
            num_records_required.min(train_size)
        };

        let dim = 384;

        let dao_meta = DaoMetaData {
            name: "Pubmed".to_string(),
            description: "PubmedHDF5Dataset".to_string(),
            data_disk_format: "h5".to_string(),
            path_to_data: data_path.to_string(),
            normed: Normed::L2,
            num_records: num_records,
            dim: dim,
        };

        let dao = JitDao::<f32> {
            _marker: Default::default(),
            meta: dao_meta,
            data,
            queries,
            num_data: num_records,
            num_queries: num_queries,
        };

        Ok(dao)
    }
}

impl JitDao<f32> {
    pub fn get_dim(&self) -> usize {
        self.meta.dim
    }

    pub fn data_len(&self) -> usize {
        self.num_data
    }

    pub fn query_len(&self) -> usize {
        self.num_queries
    }

    pub fn get_datum(&self, id: usize) -> Array1<f32> {
        if id >= self.num_data {
            panic!("id out of bounds | ID {}", id);
        }

        self.data
            .read_slice_1d(s![id, ..])
            .unwrap_or_else(|_| panic!("Cannot read slice")) //.expect("Failed to read data slice with id: {}", id) // return the row
    }

    pub fn get_query(&self, id: usize) -> Array1<f32> {
        if id >= self.num_queries {
            panic!("id out of bounds");
        }

        self.queries
            .read_slice_1d(s![id, ..])
            .unwrap_or_else(|_| panic!("Cannot read slice")) //  ?.expect("Failed to read query slice with id: {}", id) // return the row
    }
}

// #[cfg(test)]
// mod tests {
//     use crate::hdf5_dao_matrix_loader;
//     use crate::jit_dao::{hdf5_f32_jit_load, JitDao};

//     #[test]
//     fn get_query() {
//         let data_path = "/home/fm208/Downloads/datasets/pubmed/benchmark-dev-pubmed23.h5";
//         let num_queries = 10_000;
//         let all_records = 0;

//         let jit_dao: JitDao<f32> = hdf5_f32_jit_load(data_path, all_records, num_queries).unwrap();

//         let dao =
//             hdf5_dao_matrix_loader::hdf5_matrix_load(data_path, all_records, num_queries).unwrap();

//         assert_eq!(jit_dao.get_datum(2), dao.get_datum(2));
//         assert_eq!(jit_dao.get_query(2), dao.get_query(2));
//     }
// }
