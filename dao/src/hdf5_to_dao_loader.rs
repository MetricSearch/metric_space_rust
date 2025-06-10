use crate::{Dao, DaoMetaData, Normed};
use bits::{f32_embedding_to_bsp, EvpBits};
use deepsize::DeepSizeOf;
use hdf5::{File, Ix};
use ndarray::{s, Array1, Array2, ArrayBase, Ix1, OwnedRepr};
use rayon::prelude::*;
use std::cmp::min;
use tracing::error;
use utils::bytes_fmt;
use std::marker::PhantomData;

pub fn hdf5_f32_to_bsp_load(
    data_path: &str,
    num_records_required: usize, // zero if all the data
    num_queries: usize,
    num_vertices: usize,
) -> anyhow::Result<Dao<EvpBits<2>>> {
    let file = File::open(data_path)?; // open for reading
    let h5_data = file.dataset("train")?; // the data

    let train_size = h5_data.shape()[0]; // 23_887_701

    if num_records_required > train_size {
        error!("Too many records requested")
    }
    let num_records = if num_records_required == 0 {
        train_size
    } else {
        num_records_required.min(train_size)
    };

    let dim = 384;
    let mut rows_at_a_time = 5000;

    if rows_at_a_time > num_records {
        rows_at_a_time = num_records;
    }

    // Read in the file in parallel.
    // The hdf5 crate (https://docs.rs/hdf5) is not fully thread-safe when using a single handle.
    // The recommended approach in Rust is to reopen the dataset or file handle in each thread for reading.

    // 1. Set up ranges of chunks
    let chunks = (0..num_records)
        .step_by(rows_at_a_time)
        .map(|start| {
            let end = (start + rows_at_a_time).min(num_records);
            (start, end)
        })
        .collect::<Vec<(usize, usize)>>();

    let mut bsp_data: Vec<EvpBits<2>> = chunks
        .par_iter()
        .flat_map(|&(start, end)| {
            // Read slice – safe if ds_data supports concurrent reads, or re-open handle here
            let data: Array2<f32> = h5_data
                .read_slice(s![start..end, ..])
                .expect("Failed to read slice");

            data.rows()
                .into_iter()
                .map(|x| f32_embedding_to_bsp::<2>(&x, num_vertices))
                .collect::<Vec<EvpBits<2>>>()
        })
        .collect();

    log::error!("{}", bytes_fmt(bsp_data.deep_size_of()));

    // Don't bother doing this in parallel
    // Queries not big enough

    let file = File::open(data_path)?; // open for reading - this is a new open for queries only
    let o_queries_group = file.group("otest")?; // in distribution queries
    let o_queries = o_queries_group.dataset("queries").unwrap();
    let o_test_size = o_queries.shape()[0]; // 11_000
                                            // let o_queries_group = file.group("otest")?;     // out of distribution queries

    if num_queries > o_test_size {
        error!("Too many records requested")
    }
    let num_queries = if num_queries == 0 {
        o_test_size
    } else {
        num_queries.min(o_test_size)
    };

    let o_queries = o_queries_group.dataset("queries").unwrap();

    (0..num_queries).step_by(rows_at_a_time).for_each(|i| {
        let start = i;
        let end = min(i + rows_at_a_time, num_queries);

        let o_queries_slice: Array2<f32> = o_queries.read_slice(s![start..end, ..]).unwrap();

        bsp_data.extend(
            o_queries_slice
                .rows()
                .into_iter()
                .map(|x| f32_embedding_to_bsp::<2>(&x, num_vertices)),
        );
    });

    log::error!("{}", bytes_fmt(bsp_data.deep_size_of()));

    let all_combined: Array1<EvpBits<2>> = Array1::from_vec(bsp_data);

    log::error!("{}", bytes_fmt(all_combined.deep_size_of()));

    let dao_meta = DaoMetaData {
        name: "Pubmed".to_string(),
        description: "PubmedHDF5Dataset".to_string(),
        data_disk_format: "".to_string(),
        path_to_data: "".to_string(),
        normed: Normed::L2,
        num_records: num_records,
        dim: dim,
    };

    let dao = Dao {
        meta: dao_meta,
        num_data: num_records,
        num_queries: num_queries,
        embeddings: all_combined,
    };

    Ok(dao)
}

struct JitDao<T>{
    _marker: PhantomData<T>,
    meta: DaoMetaData,
    num_data: Ix,
    num_queries: usize,
}

pub fn hdf5_f32_jit_load(
    data_path: &str,
    num_records_required: usize, // zero if all the data
    num_queries: usize,
) -> anyhow::Result<JitDao<f32>> {
    let file = File::open(data_path)?; // open for reading
    let h5_data = file.dataset("train")?; // the data

    let train_size = h5_data.shape()[0];

    if num_records_required > train_size {
        error!("Too many records requested")
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
        num_data: num_records,
        num_queries: num_queries,
    };

    Ok(dao)
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

        let file = File::open(&self.meta.path_to_data).unwrap_or_else(|_| panic!("Cannot open h5 file: {}", &self.meta.path_to_data ) ); // open for reading
        let h5_data = file.dataset("train").unwrap_or_else(|_| panic!("Cannot open train dataset" ) ); // the Dataset containing the data

        println!( "Dims {:?}",h5_data.shape() );
        h5_data.read_slice_1d(s![id,..]).unwrap_or_else(|_| panic!("Cannot read slice" )) //.expect("Failed to read data slice with id: {}", id) // return the row
    }

    pub fn get_query(&self, id: usize) -> Array1<f32> {
        if id >= self.num_queries {
            panic!("id out of bounds");
        }

        let file = File::open(&self.meta.path_to_data).unwrap_or_else(|_| panic!("Cannot open h5 file: {}", &self.meta.path_to_data ) ); // open for reading
        let o_queries_group = file.group("otest").unwrap_or_else(|_| panic!("Cannot open group otest" ) ); // the group of the otest
        let o_queries = o_queries_group.dataset("queries").unwrap_or_else(|_| panic!("Cannot open queries dataset" ) ); // the Dataset containing the queries

        o_queries.read_slice_1d(s![id,..]).unwrap_or_else(|_| panic!("Cannot read slice" )) //  ?.expect("Failed to read query slice with id: {}", id) // return the row
    }
}

#[cfg(test)]
mod tests {
    use crate::Dao;
    use crate::hdf5_to_dao_loader::{hdf5_f32_jit_load, hdf5_f32_to_bsp_load, JitDao};

    #[test]
    fn get_query() {
        let source_path =  "/Volumes/Data/sisap_challenge_25/gooaq/benchmark-dev-gooaq.h5";
        const num_queries: usize = 10_000;
        const ALL_RECORDS: usize = 0;
        let dao_f32: JitDao<f32> = hdf5_f32_jit_load(&source_path, ALL_RECORDS, num_queries).unwrap();

        println!( "Data 2 : {:?}",dao_f32.get_datum(2) );
        println!( "Query 2 : {:?}", dao_f32.get_query(2) );
        panic!( "Got to here - force panic");
    }
}


