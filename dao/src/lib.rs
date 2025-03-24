// Dao impl
// al * ben

mod class_labels;
pub mod csv_f32_loader;
pub mod hdf5_f32_loader;
mod csv_nn_table_loader;
pub mod convert_f32_to_cubic;
mod convert_f32_to_cube_oct;

pub use anndists::{dist::DistDot, prelude::*};
use anyhow::Result;
use bitvec_simd::BitVecSimd;

use ndarray::{s, Array1, ArrayView1};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Read;
use wide::u64x4;

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
        assert!(a.len() == b.len());
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
        if id < self.num_data && id >= self.meta.num_records {
            panic!("id out of bounds");
        }
        self.embeddings.get(self.num_data + id).unwrap()
    }

    pub fn get_data(&self) -> ArrayView1<T> {
        let data = self.embeddings.slice(s![0..self.num_data]);
        data
    }

    pub fn get_queries(&self) -> ArrayView1<T> {
        let queries = self.embeddings.slice(s![self.num_queries..]);
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
