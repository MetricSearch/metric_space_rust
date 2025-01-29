// Dao impl
// al * ben

pub mod csv_f32_loader;
mod nn_table;
mod class_labels;

use std::fs::File;
use std::io::Read;
pub use anndists::{dist::DistDot, prelude::*};
use anyhow::{anyhow, Result};
use ndarray::{Array2, ArrayBase, Ix1, ViewRepr, Axis, ArrayView2};
use serde::{Deserialize, Serialize};
use crate::csv_f32_loader::csv_f32_loader;

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
    pub description: String,         // An English description of the data e.g. Mirflkr 1M encoded with Dino2
    pub data_disk_format: String,    // A descriptor of the data format on disk - may be used to determine the name of loader e.g format = "csv_f32" -> use the csv_f32_loader
    pub path_to_data: String,        // the path to where the data is stored on disk - URL?
    pub path_to_nns: String,         // the path to where the nns are stored on disk
    pub normed: Normed,              // is the data normed?
    pub num_records: usize,          // the total number of records/data items/rows in the data set
    pub dim: usize,                  // the dimension/number of columns in the data set
}

pub struct Dao {
    pub meta: DaoMetaData,                           // The meta data for this dao
    pub num_data: usize,                             // the size of the data (a subset of the total data)
    pub num_queries: usize,                          // the size of the queries (a subset of the total data)
    pub all_embeddings: Array2<f32>,                // the data and queries
    // pub nns: Option<Array2<usize>>,                 // the nn table (if available) TODO put elsewhere
}

impl Dao {
    pub fn new(meta_data: DaoMetaData, num_data: usize, num_queries: usize) -> Result<Self> {
        let data_and_queries: Array2<f32> = csv_f32_loader(&meta_data.path_to_data).or_else(|e| Err(anyhow!("Error loading data: {}", e)))?;

        Ok(Dao {
            meta: meta_data,
            num_data,
            num_queries,
            all_embeddings: data_and_queries,
            // nns: None,
        })
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

pub fn get_dao_metadata(meta_data_filename: &str) -> Result<DaoMetaData> {
    let mut file = File::open(meta_data_filename)?;
    let mut contents =  String::new();
    file.read_to_string(&mut contents)?;
    Ok(toml::from_str(&contents).unwrap())
}

pub fn dao_from_description(meta_data_filename: &str, num_data: usize, num_queries: usize) -> Dao {
    Dao::new(get_dao_metadata(meta_data_filename).unwrap(), num_data, num_queries).unwrap()
}
