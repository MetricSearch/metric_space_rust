use crate::{Dao, DaoMetaData, Normed};
use hdf5::{File};
use ndarray::{s, Array1, Array2};
use tracing::error;
use bits::{Bsp, f32_embedding_to_bsp};
//use tracing::metadata;

pub fn hdf5_pubmed_f32_to_bsp_load(
    data_path: &str,
    num_records_required: usize, // zero if all the data
    num_queries: usize,
    num_vertices: usize,
) -> anyhow::Result<Dao<Bsp<2>>> {
    let file = File::open(data_path)?;                   // open for reading
    let ds_data = file.dataset("train")?;       // the data
    let i_queries_group = file.group("itest")?;     // in distribution queries
    let o_queries_group = file.group("otest")?;     // out of distribution queries

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

    let mut bsp_data: Array1<Bsp<2>> = unsafe { Array1::<Bsp<2>>::uninit(num_records).assume_init()};

    for start in (0..num_records).step_by(rows_at_a_time) {
        let end = (start + rows_at_a_time).min(num_records);
        let data: Array2<f32> = ds_data.read_slice(s![start..end, ..]).unwrap();

        let bsp_rows = data
            .rows()
            .into_iter()
            .map(|x| f32_embedding_to_bsp::<2>(&x, num_vertices))
            .collect::<Array1<Bsp<2>>>();

        bsp_data
            .slice_mut(s![start..end])
            .assign(&bsp_rows);
    }

    let i_queries_slice: Array2<f32> = i_queries.read_slice(s![0..num_queries, ..]).unwrap();

    let bsp_i_test: Array1<Bsp<2>> = i_queries_slice // i_queries.read_slice(s![.., ..]).unwrap()  // read the dataset i_test queries
        .rows()
        .into_iter()
        .map(|x| f32_embedding_to_bsp::<2>(&x, num_vertices))
        .collect();

    // let o_queries_slice: Array2<f32> = o_queries.read_slice(s![.., ..]).unwrap();
    //
    // let bsp_o_test: Array1<bsp<2>> = o_queries_slice // Array1<bsp<2>> o_queries.read_slice(s![.., ..])
    //     .rows()
    //     .into_iter()
    //     .map(|x| f32_embedding_to_bsp::<2>(&x, 200))
    //     .collect();

    // let queries_combined: Array1<bsp<2>> = bsp_i_test
    //     .into_iter()
    //     .chain(bsp_o_test.into_iter())
    //     .collect();

    let all_combined: Array1<Bsp<2>> = bsp_data
        .into_iter()
        .chain(bsp_i_test.into_iter())
        .collect();

    let dao_meta = DaoMetaData {
        name: name.to_string(),
        description: description.to_string(),
        data_disk_format: "".to_string(),
        path_to_data: "".to_string(),
        normed: Normed::L2,                     // TODO <<<< wrong?
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
