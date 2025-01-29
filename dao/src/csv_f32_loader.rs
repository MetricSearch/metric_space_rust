use anyhow::anyhow;
use ndarray::{Array, Array2};


pub fn csv_f32_load(data_path: &String) -> anyhow::Result<Array2<f32>> {
    // returns a tuple stores data in single Vector
    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(b',')
        .has_headers(false)
        .from_path(data_path)?;

    let mut data_vec = Vec::new();
    let mut count = 0;
    rdr.records()
        .filter_map(|r| r.ok())
        //.take(100 ) // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<, put back in to limit
        .for_each(|r| {
            // row in the file
            data_vec.extend(r.iter().filter_map(|f| f.parse::<f32>().ok()));
            if count % 100_000 == 0 {
                tracing::info!("Ingested {count} records");
            };
            count += 1;
        });
    tracing::info!("Ingested {count} records");

    Ok(Array::from_shape_vec((count, data_vec.len() / count), data_vec)?)
}
