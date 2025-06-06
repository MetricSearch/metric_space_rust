use crate::{Dao, DaoMetaData, Normed};
use bits::{f32_embedding_to_bsp, EvpBits};
use deepsize::DeepSizeOf;
use hdf5::File;
use ndarray::{s, Array1, Array2};
use rayon::prelude::*;
use std::cmp::min;
use tracing::error;
use utils::bytes_fmt;
//use tracing::metadata;

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

pub fn hdf5_pubmed_f32_to_bsp_load_sequential(
    data_path: &str,
    num_records_required: usize, // zero if all the data
    num_queries: usize,
    num_vertices: usize,
) -> anyhow::Result<Dao<EvpBits<2>>> {
    let file = File::open(data_path)?; // open for reading
    let ds_data = file.dataset("train")?; // the data
                                          //let i_queries_group = file.group("itest")?;     // in distribution queries
    let o_queries_group = file.group("otest")?; // out of distribution queries

    let o_queries = o_queries_group.dataset("queries").unwrap();
    //let o_queries = o_queries_group.dataset("queries").unwrap();

    let train_size = ds_data.shape()[0]; // 23_887_701
    let o_test_size = o_queries.shape()[0]; // 11_000;       // used as queries
                                            //let i_test_size = i_queries.shape()[0]; // 11_000

    if num_records_required > train_size {
        error!("Too many records requested")
    }
    let num_records = if num_records_required == 0 {
        train_size
    } else {
        num_records_required.min(train_size)
    };

    if num_queries > o_test_size {
        error!("Too many records requested")
    }
    let num_queries = if num_queries == 0 {
        o_test_size
    } else {
        num_queries.min(o_test_size)
    };

    let name = "Pubmed";
    let description = "PubmedHDF5Dataset";

    let dim = 384;

    let mut rows_at_a_time = 1000;

    if rows_at_a_time > num_records {
        rows_at_a_time = num_records;
    }

    let mut bsp_data: Array1<EvpBits<2>> =
        unsafe { Array1::<EvpBits<2>>::uninit(num_records).assume_init() };

    for start in (0..num_records).step_by(rows_at_a_time) {
        let end = (start + rows_at_a_time).min(num_records);
        let data: Array2<f32> = ds_data.read_slice(s![start..end, ..]).unwrap();

        let bsp_rows = data
            .rows()
            .into_iter()
            .map(|x| f32_embedding_to_bsp::<2>(&x, num_vertices))
            .collect::<Array1<EvpBits<2>>>();

        bsp_data.slice_mut(s![start..end]).assign(&bsp_rows);
    }

    let o_queries_slice: Array2<f32> = o_queries.read_slice(s![0..num_queries, ..]).unwrap();

    let bsp_o_test: Array1<EvpBits<2>> = o_queries_slice // i_queries.read_slice(s![.., ..]).unwrap()  // read the dataset i_test queries
        .rows()
        .into_iter()
        .map(|x| f32_embedding_to_bsp::<2>(&x, num_vertices))
        .collect();

    // let queries_combined: Array1<bsp<2>> = bsp_i_test
    //     .into_iter()
    //     .chain(bsp_o_test.into_iter())
    //     .collect();

    let all_combined: Array1<EvpBits<2>> =
        bsp_data.into_iter().chain(bsp_o_test.into_iter()).collect();

    let dao_meta = DaoMetaData {
        name: name.to_string(),
        description: description.to_string(),
        data_disk_format: "".to_string(),
        path_to_data: "".to_string(),
        normed: Normed::L2, // TODO <<<< wrong?
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
