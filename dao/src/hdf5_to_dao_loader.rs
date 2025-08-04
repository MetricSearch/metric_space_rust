use crate::{Dao, DaoMetaData, Normed};
use bits::{container::BitsContainer, EvpBits};
use deepsize::DeepSizeOf;
use hdf5::{Dataset, File, Ix};
use ndarray::{s, Array1, Array2};
use rayon::prelude::*;
use std::cmp::min;
use std::path::Path;
use tracing::error;
use utils::bytes_fmt;

pub fn hdf5_f32_to_bsp_load<C: BitsContainer, const W: usize>(
    data_path: &str,
    num_records_required: usize, // zero if all the data
    num_queries: usize,
    num_vertices: usize,
) -> anyhow::Result<Dao<EvpBits<C, W>>> {
    let file = File::open(data_path)?; // open for reading
    let h5_data = file.dataset("train")?; // the data

    let data_size = h5_data.shape()[0]; // 23_887_701

    if num_records_required > data_size {
        error!("Too many records requested")
    }
    let num_records = if num_records_required == 0 {
        data_size
    } else {
        num_records_required.min(data_size)
    };

    let dim = 384;
    let mut rows_at_a_time = 5000;

    if rows_at_a_time > num_records {
        rows_at_a_time = num_records;
    }

    // Read in the file in parallel.
    // The hdf5 crate (https://docs.rs/hdf5) is not fully thread-safe when using a single handle.
    // The recommended approach in Rust is to reopen the dataset or file handle in each thread for reading.

    let mut bsp_data = parallel_read_dataset(num_vertices, h5_data, num_records, rows_at_a_time);

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
                .map(|x| EvpBits::from_embedding(x, num_vertices)),
        );
    });

    log::error!("{}", bytes_fmt(bsp_data.deep_size_of()));

    let all_combined = Array1::from_vec(bsp_data);

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
        base_addr: 0,
        num_queries: num_queries,
        embeddings: all_combined,
    };

    Ok(dao)
}

pub fn load_h5_files<C: BitsContainer, const W: usize>(
    base_path: &Path,
    filenames: &Vec<String>,
    num_vertices: usize,
    base_address: u32,
) -> anyhow::Result<Dao<EvpBits<C, W>>> {
    let mut loaded = 0;

    let mut bits = vec![];

    // load all data from the h5 files into a single Dao object.
    for data_path in filenames {
        let path = base_path.join(&data_path);
        let file = File::open(path)?; // open for reading
        let h5_data = file.dataset("data")?; // the data // TODO make a parameter.
        let data_size = h5_data.shape()[0];
        let mut bsp_data = parallel_read_dataset(num_vertices, h5_data, data_size, 5000);
        bits.append(&mut bsp_data);
        loaded += data_size;
        log::info!("loaded {} data from {}", data_size, data_path);
    }

    let embeddings = Array1::from_vec(bits);

    let dao_meta = DaoMetaData {
        name: "TODO".to_string(),        // TODO
        description: "TODO".to_string(), // TODO
        data_disk_format: "".to_string(),
        path_to_data: "".to_string(),
        normed: Normed::L2,
        num_records: loaded,
        dim: 512, // TODO
    };

    let dao = Dao {
        // TODO
        meta: dao_meta,
        num_data: loaded,
        base_addr: base_address,
        num_queries: 0,
        embeddings: embeddings,
    };

    Ok(dao)
}

fn parallel_read_dataset<C: BitsContainer, const W: usize>(
    num_vertices: usize,
    h5_data: Dataset,
    num_records: Ix,
    rows_at_a_time: Ix,
) -> Vec<EvpBits<C, W>> {
    // 1. Set up ranges of chunks
    let chunks = (0..num_records)
        .step_by(rows_at_a_time)
        .map(|start| {
            let end = (start + rows_at_a_time).min(num_records);
            (start, end)
        })
        .collect::<Vec<(usize, usize)>>();

    let bsp_data = chunks
        .par_iter()
        .flat_map(|&(start, end)| {
            // Read slice – safe if ds_data supports concurrent reads, or re-open handle here
            let data: Array2<f32> = h5_data
                .read_slice(s![start..end, ..])
                .expect("Failed to read slice");

            data.rows()
                .into_iter()
                .map(|x| EvpBits::from_embedding(x, num_vertices))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    bsp_data
}
