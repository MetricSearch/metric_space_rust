use crate::Dao;
use anyhow::anyhow;
use ndarray::Array;

pub fn csv_loader(data_path: &str, num_data: usize, num_queries: usize) -> anyhow::Result<Dao> {
    // returns a tuple stores data in single Vector
    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(b',')
        .has_headers(false)
        .from_path(data_path)?;

    let mut all_data = Vec::new();
    let mut count = 0;
    rdr.records()
        .filter_map(|r| r.ok())
        //.take(100 ) // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<, put back in to limit
        .for_each(|r| {
            // row in the file
            all_data.extend(r.iter().filter_map(|f| f.parse::<f32>().ok()));
            if count % 100_000 == 0 {
                tracing::info!("Ingested {count} records");
            };
            count += 1;
        });
    if num_data + num_queries > all_data.len() {
        Err(anyhow!("Requested data {num_data} and query {num_queries} sizes cannot be satisfied: in data of length {}", all_data.len()))
    } else {
        let dim = all_data.len() / count;

        let (data_slice, queries_slice) = all_data.split_at(num_data*dim);

        let data = Array::from_iter(data_slice.iter().map(|x| *x )).to_shape((num_data,dim)).unwrap().into_owned();
        let queries = Array::from_iter(queries_slice.iter().map(|x| *x )).to_shape((num_queries,dim)).unwrap().into_owned();

        Ok(Dao {
            num_data,
            num_queries,
            data,
            queries,
            dim,
        })
    }
}
