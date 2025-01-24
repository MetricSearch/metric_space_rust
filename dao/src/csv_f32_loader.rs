use crate::{Dao, DaoMetaData, Normed};
use anyhow::anyhow;
use ndarray::{Array, Array2};

pub fn csv_f32_loader(meta: DaoMetaData, data_path: String, num_data: usize, num_queries: usize) -> anyhow::Result<Dao> {
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

    if num_data + num_queries > count {
        Err(anyhow!("Requested data {num_data} and query {num_queries} sizes cannot be satisfied: in data of length {}", data_vec.len()))
    } else {
        let dim = data_vec.len() / count;
        let data_and_queries = Array::from_iter(data_vec.iter().map(|x| *x)).to_shape((count, dim)).unwrap().into_owned();

        Ok(Dao {
            meta,
            num_data,
            num_queries,
            all_data: data_and_queries,
            nns: None,
        })
    }
}
