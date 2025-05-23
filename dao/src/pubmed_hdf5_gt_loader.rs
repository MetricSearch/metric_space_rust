use hdf5::{File};
use ndarray::{s, Array1, Array2, Axis};

pub fn hdf5_pubmed_gt_load(
    data_path: &str,
    num_knns_required: usize,
) -> anyhow::Result<(Array2<usize>,Array2<f32>)> {
    let file = File::open(data_path)?;                   // open for reading
    let o_queries_group = file.group("otest")?;     // out of distribution queries

    let o_queries_knns = o_queries_group.dataset("knns").unwrap();  // 11000 X 1000
    let o_queries_dists = o_queries_group.dataset("dists").unwrap();
    // let o_queries_dists = o_queries_group.dataset("dists").unwrap();
    let num_rows = o_queries_knns.shape()[0] as usize;
    let num_cols = o_queries_knns.shape()[1] as usize;

    let num_cols = if num_knns_required == 0 { num_cols } else { num_knns_required.min(num_cols) } as usize;

    let knns: Array2<usize> = o_queries_knns.read_slice(s![0..num_rows, 0..num_cols]).unwrap();
    let dists: Array2<f32> = o_queries_dists.read_slice(s![0..num_rows, 0..num_cols]).unwrap();

    // Ok(filter_knns(knns,num_cols).unwrap())

    Ok((knns.slice(s![0..num_rows, 0..num_cols]).to_owned(),
       dists.slice(s![0..num_rows, 0..num_cols]).to_owned()))
}

// Filter out all entries that are greater than or equal to num_records (rows run from 0..num_records)
// then only keep the first num_cols of data
fn filter_knns(knns: Array2<usize>, num_cols: usize) -> anyhow::Result<Array2<usize>> {
    // Vector to collect cleaned rows
    let mut filtered_rows: Vec<Array1<usize>> = Vec::new();

    for row in knns.rows() {
        let filtered: Vec<usize> = row.iter()
            .take(num_cols)
            .cloned()
            .collect();

        filtered_rows.push(Array1::from(filtered));
    }

    // Stack all rows vertically to form an Array2
    let required_knns = ndarray::stack(Axis(0), &filtered_rows.iter().map(|x| x.view()).collect::<Vec<_>>())?;

    Ok(required_knns)
}