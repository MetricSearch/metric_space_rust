use crate::Dao;
use anyhow::anyhow;

pub fn csv_loader(data_path: &str, num_data: usize, num_queries: usize) -> anyhow::Result<(Dao)> {
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
        let data = all_data[0..num_data * dim].to_vec();
        let queries = all_data[num_data * dim..].to_vec();

        Ok(Dao {
            num_data,
            num_queries,
            data,
            queries,
            //nns: load_mf_nns(nn_path),
            nns: Vec::new(), // should be like above
            dim,
        })
    }
}
