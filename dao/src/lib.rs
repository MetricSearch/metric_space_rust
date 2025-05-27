// Dao impl
// al * ben

pub use anndists::{dist::DistDot, prelude::*};
use anyhow::Result;
use bits::f32_embedding_to_evp;
use bitvec_simd::BitVecSimd;
use ndarray::{s, Array1, Array2, ArrayBase, ArrayView1, ArrayView2, Ix1, ViewRepr};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::string::ToString;
use wide::u64x4;

mod class_labels;
pub mod convert_f32_to_bsp;
pub mod convert_f32_to_cube_oct;
pub mod convert_f32_to_cubic;
pub mod convert_f32_to_evp;
pub mod csv_dao_loader;
pub mod csv_dao_matrix_loader;
mod csv_nn_table_loader;
pub mod glove100_hdf5_dao_loader;
pub mod hdf5_dao_loader;
mod hdf5_dao_matrix_loader;
pub mod laion_10_m_hdf5_dao_loader;
pub mod laion_10_m_pca500_hdf5_dao_loader;
pub mod pubmed_hdf5_gt_loader;
pub mod pubmed_hdf5_to_dao_loader;
pub mod pubmed_hdf5_to_i8_dao_loader;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum Normed {
    L1,
    L2,
    None,
}

impl ToString for Normed {
    fn to_string(&self) -> String {
        match self {
            Normed::L1 => "L1".to_string(),
            Normed::L2 => "L2".to_string(),
            Normed::None => "None".to_string(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DaoMetaData {
    pub name: String,
    /// An English description of the data e.g. Mirflkr 1M encoded with Dino2
    pub description: String,
    ///  A descriptor of the data format on disk - may be used to determine the name of loader e.g format = "csv_f32" -> use the csv_f32_loader
    pub data_disk_format: String,
    /// the path to where the data is stored on disk - URL?
    pub path_to_data: String,
    /// is the data normed?
    pub normed: Normed,
    /// the total number of records/data items/rows in the data set
    pub num_records: usize,
    /// the dimension/number of columns in the data set
    pub dim: usize,
}

impl DaoMetaData {
    pub fn from_directory<P: AsRef<Path>>(dir_name: P) -> Result<Self> {
        let meta_data_file_path = dir_name.as_ref().join("meta_data.txt");

        let mut file = File::open(meta_data_file_path)?;
        let mut contents = String::new();

        file.read_to_string(&mut contents)?;

        Ok(toml::from_str(&contents).unwrap())
    }
}

pub struct Dao<Element> {
    /// The meta data for this dao
    pub meta: DaoMetaData,
    /// the size of the data (a subset of the total data)
    pub num_data: usize,
    /// the size of the queries (a subset of the total data)
    pub num_queries: usize,
    /// the data and queries
    pub embeddings: Array1<Element>,
}

impl<T> Dao<T> {
    // pub fn new(dir_name: &str) -> Self {
    //     todo!()
    // }

    pub fn get_dim(&self) -> usize {
        self.meta.dim
    }

    pub fn data_len(&self) -> usize {
        self.num_data
    }

    pub fn query_len(&self) -> usize {
        self.num_queries
    }

    pub fn get_datum(&self, id: usize) -> &T {
        if id >= self.num_data {
            panic!("id out of bounds | ID {}", id);
        }
        self.embeddings.get(id).unwrap()
    }

    pub fn get_query(&self, id: usize) -> &T {
        if id >= self.num_queries {
            panic!("id out of bounds");
        }
        self.embeddings.get(self.num_data + id).unwrap()
    }

    pub fn get_data(&self) -> ArrayView1<T> {
        let data = self.embeddings.slice(s![0..self.num_data]);
        data
    }

    pub fn get_queries(&self) -> ArrayView1<T> {
        let queries = self.embeddings.slice(s![self.num_data..]);
        queries
    }
}

pub struct DaoMatrix<T> {
    /// The meta data for this dao
    pub meta: DaoMetaData,
    /// the size of the data (a subset of the total data)
    pub num_data: usize,
    /// the size of the queries (a subset of the total data)
    pub num_queries: usize,
    /// the data and queries
    pub embeddings: Array2<T>,
}

impl<T> DaoMatrix<T> {
    pub fn new(
        meta_data: DaoMetaData,
        all_embeddings: Array2<T>,
        num_data: usize,
        num_queries: usize,
    ) -> Self {
        Self {
            meta: meta_data,
            num_data,
            num_queries,
            embeddings: all_embeddings,
        }
    }

    pub fn get_dim(&self) -> usize {
        self.meta.dim
    }

    pub fn data_len(&self) -> usize {
        self.num_data
    }

    pub fn query_len(&self) -> usize {
        self.num_queries
    }

    pub fn get_datum(&self, id: usize) -> ArrayBase<ViewRepr<&T>, Ix1> {
        if id >= self.num_data {
            // TODO consider putting option back in here
            panic!("id out of bounds");
        }
        self.embeddings.row(id)
    }

    pub fn get_query(&self, id: usize) -> ArrayBase<ViewRepr<&T>, Ix1> {
        if id < self.num_data && id >= self.meta.num_records {
            // TODO consider putting option back in here
            panic!("id out of bounds");
        }
        self.embeddings.row(self.num_data + id)
    }

    pub fn get_data(&self) -> ArrayView2<T> {
        self.embeddings.slice(s![..self.num_data, ..])
    }

    pub fn get_queries(&self) -> ArrayView2<T> {
        self.embeddings.slice(s![self.num_data.., ..])
    }
}

// TODO FIX AND MOVE THESE
// pub fn dao_from_h5(data_path: &str, num_data: usize, num_queries: usize) -> std::result::Result<DaoMatrix> {
//     hdf5_f32_load(data_path, num_data, num_queries)
// }
//
// pub fn dao_from_csv_dir(dir_name: &str, num_data: usize, num_queries: usize) -> std::result::Result<DaoMatrix> {
//     let meta_data = dao_metadata_from_dir(dir_name).unwrap();
//     let mut data_file_path = dir_name.to_string();
//     data_file_path.push_str("/");
//     data_file_path.push_str(meta_data.path_to_data.as_str());
//
//     // Put loader selection here.
//
//     let data_and_queries: Array2<f32> = csv_f32_load(&data_file_path).or_else(|e| Err(anyhow!("Error loading data: {}", e)))?;
//
//     Ok(Self::new(meta_data, data_and_queries, num_data, num_queries))
// }

pub fn f32_embeddings_to_evp<const D: usize>(
    embeddings: &Array1<Array1<f32>>,
    non_zeros: usize,
) -> Array1<BitVecSimd<[u64x4; D], 4>> {
    embeddings
        .iter()
        .map(|row| f32_embedding_to_evp::<D>(row, non_zeros))
        .collect()
}
