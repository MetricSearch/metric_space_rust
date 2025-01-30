// Dao impl
// al * ben

pub mod csv_f32_loader;
mod nn_table;
mod class_labels;
pub mod hdf5_f32_loader;

use std::collections::LinkedList;
use std::fs::File;
use std::io::Read;
use std::ops::{Add, Mul};
use std::str::FromStr;
pub use anndists::{dist::DistDot, prelude::*};
use anyhow::{anyhow, Result};
use ndarray::{Array2, ArrayBase, Ix1, ViewRepr, Axis, ArrayView2, Array};
use serde::{Deserialize, Serialize};
use crate::csv_f32_loader::csv_f32_load;
use crate::hdf5_f32_loader::hdf5_f32_load;

#[derive(Debug,Serialize,Deserialize,Clone)]
pub enum Normed {
    L1,
    L2,
    None
}

// impl Normed {
//     fn as_str(&self) -> &'static str {
//         match self {
//             Normed::L1 => "L1",
//             Normed::L2 => "L2",
//             _ => "none",
//         }
//     }
// }

#[derive(Debug,Serialize,Deserialize,Clone)]
pub struct DaoMetaData {
    pub name: String,
    pub description: String,         // An English description of the data e.g. Mirflkr 1M encoded with Dino2
    pub data_disk_format: String,    // A descriptor of the data format on disk - may be used to determine the name of loader e.g format = "csv_f32" -> use the csv_f32_loader
    pub path_to_data: String,        // the path to where the data is stored on disk - URL?
    pub normed: Normed,              // is the data normed?
    pub num_records: usize,          // the total number of records/data items/rows in the data set
    pub dim: usize,                  // the dimension/number of columns in the data set
}

pub struct Dao32 {
    pub meta: DaoMetaData,                           // The meta data for this dao
    pub num_data: usize,                             // the size of the data (a subset of the total data)
    pub num_queries: usize,                          // the size of the queries (a subset of the total data)
    pub all_embeddings: Array2<f32>,                // the data and queries
    // pub nns: Option<Array2<usize>>,                 // the nn table (if available) TODO put elsewhere
}

impl Dao32 {

    pub fn new(meta_data: DaoMetaData, all_embeddings: Array2<f32>, num_data: usize, num_queries: usize) -> Self {
        Self{
            meta: meta_data,
            num_data,
            num_queries,
            all_embeddings: all_embeddings,
        }
    }

    pub fn dao_from_h5(data_path: &str, num_data: usize, num_queries: usize) -> Result<Dao32> {
        hdf5_f32_load(data_path, num_data, num_queries)
    }

    pub fn dao_from_csv_dir(dir_name: &str, num_data: usize, num_queries: usize) -> Result<Dao32> {
        let meta_data = dao_metadata_from_dir(dir_name).unwrap();
        let mut data_file_path = dir_name.to_string();
        data_file_path.push_str("/");
        data_file_path.push_str(meta_data.path_to_data.as_str());

        // Put loader selection here.

        let data_and_queries: Array2<f32> = csv_f32_load(&data_file_path).or_else(|e| Err(anyhow!("Error loading data: {}", e)))?;

        Ok(Self::new(meta_data, data_and_queries, num_data, num_queries))
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
        self.all_embeddings.row(id)
    }

    pub fn get_query(&self, id: usize) -> ArrayBase<ViewRepr<&f32>,Ix1>  {
        if id < self.num_data && id >= self.meta.num_records {   // TODO consider putting option back in here
            panic!("id out of bounds");
        }
        self.all_embeddings.row(self.num_data+id)
    }

    pub fn get_data(&self) -> ArrayView2<f32> {
        let (data , _) = self.all_embeddings.view().split_at(Axis(0),self.num_data * self.meta.dim);
        data
    }

    pub fn get_queries(&self) -> ArrayView2<f32> {
        let (_ , queries) = self.all_embeddings.view().split_at(Axis(0),self.num_data * self.meta.dim);
        queries
    }
}

pub fn dao_metadata_from_dir(dir_name: &str) -> Result<DaoMetaData> {
    let mut meta_data_file_path = dir_name.to_string();
    meta_data_file_path.push_str("/meta_data.txt");
    let mut file = File::open(meta_data_file_path)?;
    let mut contents =  String::new();
    file.read_to_string(&mut contents)?;
    Ok(toml::from_str(&contents).unwrap())
}



