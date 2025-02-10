use anyhow::anyhow;
use ndarray::{Array, Array1, Array2};
use crate::{dao_metadata_from_dir, Dao};

pub fn csv_f32_load(data_path: &String) -> anyhow::Result<Array1<Array1<f32>>> {
    // returns a tuple stores data in single Vector
    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(b',')
        .has_headers(false)
        .from_path(data_path)?;

    let mut data_vec: Vec<Array1<f32>> = Vec::new();
    let mut count = 0;
    rdr.records()
        .filter_map(|r| r.ok())
        //.take(100 ) // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<, put back in to limit
        .for_each(|r| {
            // row in the file
            data_vec.push(r
                .iter()
                .filter_map(|f| f.parse::<f32>().ok()).collect::<Array1<f32>>());
            if count % 100_000 == 0 {
                tracing::info!("Ingested {count} records");
            };
            count += 1;
        });
    tracing::info!("Ingested {count} records");

    Ok(Array1::from_vec(data_vec))
    // Ok(Array::from_shape_vec(
    //     (count, data_vec.len()),
    //     data_vec,
    // )?)
}

pub fn dao_from_csv_dir(dir_name: &str, num_data: usize, num_queries: usize) -> anyhow::Result<Dao<Array1<f32>>> {
    let meta = dao_metadata_from_dir(dir_name).unwrap();
    let mut data_file_path = dir_name.to_string();
    data_file_path.push_str("/");
    data_file_path.push_str(meta.path_to_data.as_str());

    // Put loader selection here.

    let embeddings: Array1<Array1<f32>> = csv_f32_load(&data_file_path).or_else(|e| Err(anyhow!("Error loading data: {}", e)))?;

    Ok(Dao::<Array1<f32>> { meta, num_data, num_queries, embeddings } )
}
