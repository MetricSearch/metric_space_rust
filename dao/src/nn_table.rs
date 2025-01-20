pub fn load_nn_table(nn_path: &str) -> anyhow::Result<Vec<Vec<usize>>> {
    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(b',')
        .has_headers(false)
        .from_path(nn_path)?;

    let vecs: Vec<Vec<usize>> = rdr
        .records()
        .filter_map(|record| record.ok())
        .map(|record| {
            record
                .iter()
                .filter_map(|f| f.parse::<usize>().ok())
                .collect::<Vec<usize>>()
        })
        .collect::<Vec<Vec<usize>>>();

    Ok(vecs)
}
