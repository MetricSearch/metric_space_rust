// Dao impl
// al * ben

pub mod csv_loader;
mod nn_table;
mod class_labels;

pub use anndists::{dist::DistDot, prelude::*};
use anyhow::{anyhow, Result};
use rayon::prelude::*;

#[derive(Debug)]
pub struct Dao {
    pub num_data: usize,
    pub num_queries: usize,
    pub data: Vec<f32>,
    queries: Vec<f32>,
    nns: Vec<Vec<usize>>,
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

    pub fn get(&self, id: usize) -> Result<&[f32]> {
        if id > self.num_data {
            Err(anyhow!("Data index out of bounds requested {id} data range is 0..{}",self.num_data))
        } else {
            Ok(&self.data.as_slice()[id * self.dim..(id + 1) * self.dim])
        }
    }

    pub fn query(&self, id: usize) -> Result<&[f32]> {
        if id > self.num_queries {
            Err(anyhow!("Data index out of bounds requested {id} data range is 0..{}",self.num_queries))
        } else {
            Ok(&self.queries.as_slice()[id*self.dim .. (id+1) *self.dim])
        }
    }
}
