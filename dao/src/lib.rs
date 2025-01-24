// Dao impl
// al * ben

pub mod csv_f32_loader;
mod nn_table;
mod class_labels;

use std::fs::File;
use std::io::Read;
pub use anndists::{dist::DistDot, prelude::*};
use anyhow::{anyhow, Result};
use ndarray::{Array, Array1, Array2, ArrayBase, Ix1, Ix2, ViewRepr};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use toml::de::Error;
use crate::csv_f32_loader::csv_f32_loader;

#[derive(Debug,Serialize,Deserialize,Clone)]
pub enum Normed {
    L1,
    L2,
    None
}

impl Normed {
    fn as_str(&self) -> &'static str {
        match self {
            Normed::L1 => "L1",
            Normed::L2 => "L2",
            other => "none",
        }
    }
}

#[derive(Debug,Serialize,Deserialize,Clone)]
pub struct DaoMetaData {
    pub description: String,         // An English description of the data e.g. Mirflkr 1M encoded with Dino2
    pub data_disk_format: String,    // A descriptor of the data format on disk - may be used to determine the name of loader e.g format = "csv_f32" -> use the csv_f32_loader
    pub nns_disk_format: String,     // A descriptor of the nns format on disk - may be used to determine the name of loader e.g format = "csv_usize" -> use the csv_usize_loader
    pub path_to_data: String,        // the path to where the data is stored on disk - URL?
    pub path_to_nns: String,         // the path to where the nns are stored on disk
    pub normed: Normed,              // is the data normed?
    pub num_records: usize,          // the total number of records/data items/rows in the data set
    pub dim: usize,                  // the dimension/number of columns in the data set
}


#[derive(Debug)]
pub struct Dao {
    pub meta: DaoMetaData,           // The meta data for this dao
    pub num_data: usize,             // the size of the data (a subset of the total data)
    pub num_queries: usize,          // the size of the queries (a subset of the total data)
    pub all_data: Array2<f32>,       // the data and queries
    pub nns: Option<Array2<usize>>,  // the nn table (if available)
}

impl Dao {
    pub fn new(meta: DaoMetaData, num_data: usize, num_queries: usize) -> Result<Self> {
        match meta.data_disk_format.as_str() {
            "csv_f32"  => Ok(csv_f32_loader(meta.clone(), meta.path_to_data, num_data, num_queries)?),
            other => Err(anyhow!("Unsupported data disk format: {}", other))
        }
    }

    pub fn get_dim(&self) -> usize { self.meta.dim }

    pub fn data_len(&self) -> usize {
        self.num_data
    }

    pub fn query_len(&self) -> usize {
        self.num_queries
    }

    pub fn get(&self, id: usize) -> ArrayBase<ViewRepr<&f32>,Ix1> {
        if( id >= self.num_data ) { // TODO consider putting option back in here
            panic!("id out of bounds");
        }
        self.all_data.row(id)
    }

    pub fn query(&self, id: usize) -> ArrayBase<ViewRepr<&f32>,Ix1>  {
        if( id < self.num_data && id >= self.meta.num_records ) {   // TODO consider putting option back in here
            panic!("id out of bounds");
        }
        self.all_data.row(self.num_data+id)
    }
}

pub fn getDaoMetaData(meta_data_filename: &str) -> Result<DaoMetaData> {
    let mut file = File::open(meta_data_filename)?;
    let mut contents =  String::new();
    file.read_to_string(&mut contents)?;
    Ok(toml::from_str(&contents).unwrap())
}

pub fn dao_from_description(meta_data_filename: &str, num_data: usize, num_queries: usize) -> Dao {
    Dao::new(getDaoMetaData(meta_data_filename).unwrap(), num_data, num_queries).unwrap()
}
