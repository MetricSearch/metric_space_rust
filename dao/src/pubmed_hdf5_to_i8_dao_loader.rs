use crate::{DaoMatrix, DaoMetaData, Normed};
use hdf5::{File};
use ndarray::{concatenate, s, stack, Array1, Array2, Axis};
use tracing::error;
use bits::f32_embedding_to_i8_embedding;
//use tracing::metadata;

pub fn hdf5_pubmed_f32_to_i8_load(
    data_path: &str,
    num_records_required: usize, // zero if all the data
    num_queries: usize,
    num_vertices: usize,
) -> anyhow::Result<DaoMatrix<i8>> {
    let file = File::open(data_path)?;                   // open for reading
    let ds_data = file.dataset("train")?;       // the data
    let i_queries_group = file.group("itest")?;     // in distribution queries
    // let o_queries_group = file.group("otest")?;     // out of distribution queries

    let i_queries = i_queries_group.dataset("queries").unwrap();
    //let o_queries = o_queries_group.dataset("queries").unwrap();

    let train_size =  ds_data.shape()[0]; // 23_887_701
    let i_test_size = i_queries.shape()[0]; // 11_000;       // used as queries
    //let o_test_size = i_queries.shape()[0]; // 11_000

    if num_records_required > train_size {
        error!("Too many records requested" )
    }
    let num_records = if num_records_required == 0 { train_size } else { num_records_required.min(train_size) };

    if num_queries > i_test_size {
        error!("Too many records requested" )
    }
    let num_queries = if num_queries == 0 { i_test_size } else { num_queries.min(i_test_size) };

    let name = "Pubmed";
    let description ="PubmedHDF5Dataset";

    let dim = 100;

    let mut rows_at_a_time = 1000;

    if rows_at_a_time > num_records {
        rows_at_a_time = num_records;
    }

    let mut i8_data: Array2<i8> = Array2::zeros((num_records, 384));

    for start in (0..num_records).step_by(rows_at_a_time) {
        let end = (start + rows_at_a_time).min(num_records);
        let data: Array2<f32> = ds_data.read_slice(s![start..end, ..]).unwrap();

        let i_8_rows_vec: Vec<Array1<i8>> = data
            .rows()
            .into_iter()
            .map(|x| f32_embedding_to_i8_embedding(&x, num_vertices))
            .collect();

        // Stack rows vertically into a 2D array
        let i_8_array2: Array2<i8> = stack(Axis(0), &i_8_rows_vec.iter().map(|r| r.view()).collect::<Vec<_>>()).unwrap();

        i8_data
            .slice_mut(s![start..end,0..])
            .assign(&i_8_array2);
    }

    let i_queries_slice: Array2<f32> = i_queries.read_slice(s![0..num_queries, ..]).unwrap();

    let i_8_i_test_vec: Vec<Array1<i8>> = i_queries_slice
        .rows()
        .into_iter()
        .map(|x| f32_embedding_to_i8_embedding(&x, num_vertices))
        .collect();

    let i_8_i_test_array: Array2<i8> = stack(Axis(0), &i_8_i_test_vec.iter().map(|r| r.view()).collect::<Vec<_>>()).unwrap();

    let all_combined: Array2<i8> = concatenate(Axis(0), &[i8_data.view(), i_8_i_test_array.view()]).unwrap();

    let dao_meta = DaoMetaData {
        name: name.to_string(),
        description: description.to_string(),
        data_disk_format: "".to_string(),
        path_to_data: "".to_string(),
        normed: Normed::L2,                     // TODO <<<< wrong?
        num_records: num_records,
        dim: dim,
    };

    let dao = DaoMatrix::<i8> {
        meta: dao_meta,
        num_data: num_records,
        num_queries: num_queries,
        embeddings: all_combined,
    };

    Ok(dao)
}
