use crate::{DaoMatrix, DaoMetaData};
use anyhow::anyhow;
use ndarray::{Array, Array2};

pub fn csv_dao_matrix_load(data_path: &String) -> anyhow::Result<Array2<f32>> {
    // returns a tuple stores data in single Vector
    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(b',')
        .has_headers(false)
        .from_path(data_path)?;

    let mut data_vec: Vec<f32> = Vec::new();
    let mut count = 0;
    let mut rows = 0;
    rdr.records()
        .filter_map(|r| r.ok())
        //.take(100 ) // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<, put back in to limit
        .for_each(|record| {
            // row in the file
            rows = rows + 1;
            record
                .iter()
                .filter_map(|float_str| float_str.parse::<f32>().ok())
                .for_each(|float| data_vec.push(float));
            if count % 100_000 == 0 {
                tracing::info!("Ingested {count} records");
            };
            count += 1;
        });
    tracing::info!("Ingested {count} records");

    Ok(Array::from_shape_vec(
        (rows, data_vec.len() / rows),
        data_vec,
    )?)
}

pub fn dao_matrix_from_csv_dir(
    dir_name: &str,
    num_data: usize,
    num_queries: usize,
) -> anyhow::Result<DaoMatrix<f32>> {
    let meta = DaoMetaData::from_directory(dir_name).unwrap();
    let mut data_file_path = dir_name.to_string();
    data_file_path.push_str("/");
    data_file_path.push_str(meta.path_to_data.as_str());

    let embeddings: Array2<f32> = csv_dao_matrix_load(&data_file_path)
        .or_else(|e| Err(anyhow!("Error loading data: {}", e)))?;

    Ok(DaoMatrix {
        meta,
        num_data,
        num_queries,
        embeddings,
    })
}
