// Dao impl
// al * ben

mod class_labels;
pub mod csv_dao_loader;
pub mod hdf5_dao_loader;
mod csv_nn_table_loader;
pub mod convert_f32_to_cubic;
pub mod convert_f32_to_cube_oct;
mod csv_dao_matrix_loader;
mod hdf5_dao_matrix_loader;

pub use anndists::{dist::DistDot, prelude::*};
use anyhow::{Result};
use bitvec_simd::BitVecSimd;

use ndarray::{s, Array1, Array2, ArrayBase, ArrayView1, ArrayView2, Axis, Ix1, ViewRepr};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Read;
use wide::u64x4;
use crate::csv_dao_loader::csv_f32_load;
use crate::hdf5_dao_loader::hdf5_f32_load;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum Normed {
    L1,
    L2,
    None,
}

pub trait DataType {
    fn dot_product(a: &Self, b: &Self) -> f32;
    fn dist(a: &Self, b: &Self) -> f32;
}


impl DataType for Array1<f32> {
    fn dot_product(a: &Self, b: &Self) -> f32 {
        debug_assert!(a.len() == b.len());
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()

        // equivalent to:
        // let mut result = 0.0;
        // for i in 0..p0.len() {
        //     result += p0[i] * p1[i];
        // }
        // result
    }

    /// This is Euc dist
    fn dist(a: &Self, b: &Self) -> f32 {
        debug_assert!(a.len() == b.len());
        f32::sqrt(a.iter().zip(b.iter()).map(|(a, b)| (a - b).powi(2)).sum())
    }
}

impl DataType for BitVecSimd<[u64x4; 4], 4> {
    fn dot_product(p0: &Self, p1: &Self) -> f32 {
        p0.and_cloned(p1).count_ones() as f32
    }

    fn dist(a: &Self, b: &Self) -> f32 {
        a.xor_cloned(b).count_ones() as f32
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DaoMetaData {
    pub name: String,
    pub description: String, // An English description of the data e.g. Mirflkr 1M encoded with Dino2
    pub data_disk_format: String, // A descriptor of the data format on disk - may be used to determine the name of loader e.g format = "csv_f32" -> use the csv_f32_loader
    pub path_to_data: String,     // the path to where the data is stored on disk - URL?
    pub normed: Normed,           // is the data normed?
    pub num_records: usize,       // the total number of records/data items/rows in the data set
    pub dim: usize,               // the dimension/number of columns in the data set
}

pub struct Dao<DataRep: Clone> {
    pub meta: DaoMetaData,           // The meta data for this dao
    pub num_data: usize,             // the size of the data (a subset of the total data)
    pub num_queries: usize,          // the size of the queries (a subset of the total data)
    pub embeddings: Array1<DataRep>, // the data and queries
}

pub struct DaoMatrix {
    pub meta: DaoMetaData,                           // The meta data for this dao
    pub num_data: usize,                             // the size of the data (a subset of the total data)
    pub num_queries: usize,                          // the size of the queries (a subset of the total data)
    pub embeddings: Array2<f32>,                     // the data and queries
}

impl<T: Clone + DataType> Dao<T> {
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
            panic!("id out of bounds");
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

impl DaoMatrix {

    pub fn new(meta_data: DaoMetaData, all_embeddings: Array2<f32>, num_data: usize, num_queries: usize) -> Self {
        Self{
            meta: meta_data,
            num_data,
            num_queries,
            embeddings: all_embeddings,
        }
    }

    pub fn get_dim(&self) -> usize { self.meta.dim }

    pub fn data_len(&self) -> usize {
        self.num_data
    }

    pub fn query_len(&self) -> usize {
        self.num_queries
    }

    pub fn get_datum(&self, id: usize) -> ArrayBase<ViewRepr<&f32>,Ix1> {
        if id >= self.num_data  { // TODO consider putting option back in here
            panic!("id out of bounds");
        }
        self.embeddings.row(id)
    }

    pub fn get_query(&self, id: usize) -> ArrayBase<ViewRepr<&f32>,Ix1>  {
        if id < self.num_data && id >= self.meta.num_records {   // TODO consider putting option back in here
            panic!("id out of bounds");
        }
        self.embeddings.row(self.num_data+id)
    }

    pub fn get_data(&self) -> ArrayView2<f32> {
        let (data , _) = self.embeddings.view().split_at(Axis(0), self.num_data * self.meta.dim);
        data
    }

    pub fn get_queries(&self) -> ArrayView2<f32> {
        let (_ , queries) = self.embeddings.view().split_at(Axis(0), self.num_data * self.meta.dim);
        queries
    }
}


pub fn dao_metadata_from_dir(dir_name: &str) -> Result<DaoMetaData> {
    let mut meta_data_file_path = dir_name.to_string();
    meta_data_file_path.push_str("/meta_data.txt");
    let mut file = File::open(meta_data_file_path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    Ok(toml::from_str(&contents).unwrap())
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
