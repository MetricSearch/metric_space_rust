// Dao impl
// al * ben

use anndists::{dist::DistDot, prelude::*};
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
    pub fn new(
        data_path: &str,
        nn_path: &str,
        num_data: usize,
        num_queries: usize,
    ) -> Result<Self> {
        let (data, queries, dim) = load_mf_dino2(data_path, num_data, num_queries)?;
        Ok(Self {
            num_data,
            num_queries,
            data,
            queries,
            //nns: load_mf_nns(nn_path),
            nns: Vec::new(), // should be like above
            dim,
        })
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
            Err(anyhow!(
                "Data index out of bounds requested {id} data range is 0..{}",
                self.num_data
            ))
        } else {
            Ok(&self.data.as_slice()[id * self.dim..(id + 1) * self.dim])
        }
    }

    pub fn query(&self, id: usize) -> Result<&[f32]> {
        if id > self.num_queries {
            Err(anyhow!(
                "Data index out of bounds requested {id} data range is 0..{}",
                self.num_queries
            ))
        } else {
            Ok(&self.queries.as_slice()[id * self.dim..(id + 1) * self.dim])
        }
    }
}

pub fn load_mf_dino2(
    data_path: &str,
    num_data: usize,
    num_queries: usize,
) -> Result<(Vec<f32>, Vec<f32>, usize)> {
    // returns a tuple stores data in single Vector
    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(b',')
        .has_headers(false)
        .from_path(data_path)?;

    let mut all_data = Vec::new();
    let mut count = 0;
    rdr.records()
        .filter_map(|r| r.ok())
        //.take(100 ) // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<, put back in to limit
        .for_each(|r| {
            // row in the file
            all_data.extend(r.iter().filter_map(|f| f.parse::<f32>().ok()));
            if count % 100_000 == 0 {
                tracing::info!("Ingested {count} records");
            };
            count += 1;
        });
    if num_data + num_queries > all_data.len() {
        Err(anyhow!("Requested data {num_data} and query {num_queries} sizes cannot be satisfied: in data of length {}", all_data.len()))
    } else {
        let dim = all_data.len() / count;
        let data = all_data[0..num_data * dim].to_vec();
        let queries = all_data[num_data * dim..].to_vec();

        Ok((data, queries, dim))
    }
}

pub fn load_nn_table(nn_path: &str) -> Result<Vec<Vec<usize>>> {
    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(b',')
        .has_headers(false)
        .from_path(nn_path)?;

    let vecs: Vec<Vec<usize>> = rdr
        .records()
        .filter_map(|record| record.ok())
        .map(|record| {
            record
                .iter()
                .filter_map(|f| f.parse::<usize>().ok())
                .collect::<Vec<usize>>()
        })
        .collect::<Vec<Vec<usize>>>();

    Ok(vecs)
}

pub fn get_class_labels(
    hyperplane: &Vec<f32>,
    vectors: &Vec<Vec<f32>>,
    nn_table: &Vec<Vec<usize>>,
    alpha: u16,
) -> Vec<i32> {
    let dot_prod_over_data = vectors
        .par_iter()
        .map(|x| 1.0 - DistDot.eval(x.as_slice(), hyperplane.as_slice()))
        .collect::<Vec<f32>>();

    // Vector (indexed by ID) of 100NN vectors
    nn_table
        .par_iter()
        .map(|indexes| indexes.iter().filter_map(|&index| vectors.get(index))) // Maps NN indexes to vectors
        .map(|nn_vecs| nn_vecs.map(|x| 1.0 - DistDot.eval(x.as_slice(), hyperplane.as_slice()))) // Maps vectors to list of dot products
        .enumerate()
        .map(|(i, dots)| {
            dots.filter_map(|x| {
                if (x > 0.0) == (*dot_prod_over_data.get(i).unwrap() > 0.0) {
                    Some(1)
                } else {
                    None
                }
            }) // Counts the number on same side of HP as original item
            .sum::<u16>()
        })
        .map(|x| if x > alpha { 1 } else { 0 }) // Whether the sum is gt alpha
        .collect::<Vec<i32>>()
}
