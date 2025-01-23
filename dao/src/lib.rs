// Dao impl
// al * ben

pub mod csv_loader;
mod nn_table;
mod class_labels;

pub use anndists::{dist::DistDot, prelude::*};
use anyhow::{anyhow, Result};
use ndarray::{Array, Array1, Array2, Ix1, Ix2};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

//#[derive(Debug,Serialize,Deserialize)]
pub struct Dao {
    pub num_data: usize,
    pub num_queries: usize,
    pub data: Array2<f32>,
    pub queries: Array2<f32>,
    dim: usize,
}

impl Dao {
    pub fn new( data_path: &str, num_data: usize, num_queries: usize, loader: &dyn Fn(&str,usize,usize) -> anyhow::Result<Dao>) -> Result<Self> {
        loader(data_path,num_data,num_queries)
    }

    pub fn get_dim(&self) -> usize {
        self.dim
    }

    pub fn data_len(&self) -> usize {
        self.num_data
    }

    pub fn query_len(&self) -> usize {
        self.num_queries
    }

    pub fn get(&self, id: usize) -> Array1<f32> { // TODO this should return a view?
        self.data.row(id).into_owned()   // this costs us!! TODO
    }

    pub fn query(&self, id: usize) -> Array1<f32> {
        self.queries.row(id).into_owned()
    }
}
