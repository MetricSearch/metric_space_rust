use anyhow::bail;
use bits::{f32_embedding_to_bsp, EvpBits};
use memmap2::MmapOptions;
use ndarray::Array1;
use rayon::{iter::ParallelIterator as _, slice::ParallelSlice};
use std::{cmp::min, fs::File, path::Path};

use crate::{Dao, DaoMetaData, Normed};

const ROW_GROUP_SIZE: usize = 1_000_000;
const NUM_VERTICES: usize = 200;
const WIDTH: usize = 384;

pub fn load(
    data_path: &str,
    num_records_required: usize, // zero if all the data
    num_queries: usize,
    num_vertices: usize,
) -> anyhow::Result<Dao<EvpBits<2>>> {
    let file = hdf5::File::open(data_path)?; // open for reading
    let h5_data = file.dataset("train")?; // the data

    let offset = h5_data.offset().unwrap();
    let length = h5_data.size();

    let mut bsp_data = load_from_file::<WIDTH, _>(data_path, offset, length)?;

    let train_size = usize::try_from(h5_data.shape()[0]).unwrap(); // 23_887_701
    let dim = usize::try_from(h5_data.shape()[1]).unwrap();
    assert_eq!(dim, WIDTH);

    if num_records_required > train_size {
        panic!("Too many records requested")
    }
    let num_records = if num_records_required == 0 {
        train_size
    } else {
        num_records_required.min(train_size)
    };

    let o_queries_group = file.group("otest")?; // in distribution queries
    let o_queries = o_queries_group.dataset("queries").unwrap();
    let o_test_size = o_queries.shape()[0]; // 11_000
                                            // let o_queries_group = file.group("otest")?;     // out of distribution queries

    if num_queries > o_test_size {
        bail!("Too many records requested")
    }

    let num_queries = if num_queries == 0 {
        o_test_size
    } else {
        num_queries.min(o_test_size)
    };

    let offset = o_queries.offset().unwrap();
    let length = o_queries.size();

    let bsp_o_test = load_from_file::<WIDTH, _>(data_path, offset, length)?;

    let name = "Pubmed";
    let description = "PubmedHDF5Dataset";

    bsp_data.extend(bsp_o_test);

    let all_combined: Array1<EvpBits<2>> = Array1::from_vec(bsp_data);

    let dao_meta = DaoMetaData {
        name: name.to_string(),
        description: description.to_string(),
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

// fn extract_chunk_location() {
//     // get offset and size of the chunk
//     let (chunk_file_start, chunk_file_end, num_rows, num_cols) = {
//         let i = Index::index(&path)?;
//         let ds = i
//             .dataset(dataset)
//             .ok_or(anyhow::anyhow!("unknown dataset {dataset:?}"))?;
//         let DatasetD::D2(ds) = ds else { panic!() };

//         let num_rows = usize::try_from(ds.shape[0]).unwrap();
//         let num_cols = usize::try_from(ds.shape[1]).unwrap();

//         ensure!(
//             ds.chunks.len() == 1,
//             "more than one chunk in dataset {dataset:?} ({:?})",
//             path.as_ref()
//         );

//         let chunk = &ds.chunks[0];

//         (
//             chunk.addr.get(),
//             chunk.addr.get() + chunk.size.get(),
//             num_rows,
//             num_cols,
//         )
//     };
// }

pub fn load_from_file<const WIDTH: usize, P: AsRef<Path>>(
    path: P,
    file_offset: u64,
    length: usize,
) -> anyhow::Result<Vec<EvpBits<2>>> {
    let file = File::open(path)?;

    let end = file_offset + u64::try_from(length).unwrap();

    Ok((file_offset..end)
        .step_by(ROW_GROUP_SIZE * 384)
        .map(|start| {
            let end = min(start + (ROW_GROUP_SIZE * WIDTH) as u64, end);

            let map = unsafe {
                MmapOptions::new()
                    .offset(start)
                    .len((end - start) as usize)
                    .map(&file)
            }
            .unwrap();
            map.advise(memmap2::Advice::Sequential).unwrap();

            let (prefix, shorts, suffix) = unsafe { map.align_to::<f32>() };

            assert!(prefix.is_empty());
            assert!(suffix.is_empty());

            shorts
                .par_chunks(WIDTH)
                .map(|row| f32_embedding_to_bsp::<2>(row, NUM_VERTICES))
                .collect::<Vec<_>>()
        })
        .flatten()
        .collect())
}
