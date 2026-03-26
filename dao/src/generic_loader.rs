use hdf5::{File, H5Type};
use ndarray::{s, Array2, ArrayView1};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::path::Path;

/// Generic loader of HDF5 datasets
///
/// Reading and encoding to be performed in parallel.
///
/// # Parameters
/// * `path`: Path to the .hdf5 file
/// * `dataset`: Dataset within the HDF5 file to load from
/// * `record_limit`: Maximum number of records to be retrieved, `None` for all records
/// * `chunk_size`: Number of records per parallel work unit
/// * `encoder`: Function which compresses/embeds the source values of type `S` into some type `T`
pub fn par_load<
    P: AsRef<Path>,
    S: H5Type,
    T: Send,
    F: Fn(ArrayView1<S>) -> T + Send + Sync + Copy,
>(
    path: P,
    dataset: &str,
    record_limit: Option<usize>,
    chunk_size: usize,
    encoder: F,
) -> anyhow::Result<Vec<T>> {
    let file = File::open(path)?;

    let data = file.dataset(dataset)?;

    let data_size = data.shape()[0];

    let num_records = if let Some(record_limit) = record_limit {
        if record_limit > data_size {
            return Err(anyhow::anyhow!(
                "record limit ({record_limit}) greater than data size ({data_size})"
            ));
        }

        record_limit
    } else {
        data_size
    };

    // should be significantly smaller anyway, maybe warn if >8x larger?
    if chunk_size > num_records {
        return Err(anyhow::anyhow!(
            "chunk size ({chunk_size}) greater than number of records being retrieved ({num_records})"
        ));
    }

    // start and end indices for each chunk
    let chunks = (0..num_records)
        .step_by(chunk_size)
        .map(|start| {
            let end = (start + chunk_size).min(num_records);
            (start, end)
        })
        // allocated ahead of time for some reason I forget
        .collect::<Vec<(usize, usize)>>();

    Ok(chunks
        .par_iter()
        .flat_map(|&(start, end)| {
            // Read slice – safe if ds_data supports concurrent reads, or re-open handle here
            let data: Array2<S> = data
                .read_slice(s![start..end, ..])
                .expect("Failed to read slice");

            data.rows().into_iter().map(encoder).collect::<Vec<T>>()
        })
        .collect::<Vec<T>>())
}
