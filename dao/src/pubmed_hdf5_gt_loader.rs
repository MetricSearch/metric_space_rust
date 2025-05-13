use crate::{Dao, DaoMetaData, Normed};
use hdf5::{File};
use ndarray::{s, Array1, Array2, Axis};
use tracing::error;
use bits::{Bsp, f32_embedding_to_bsp};
//use tracing::metadata;

pub fn hdf5_pubmed_gt_load(
    data_path: &str,
    num_records_required: usize, // zero if all the data
    num_knns_required: usize,
) -> anyhow::Result<Array2<usize>> {
    let file = File::open(data_path)?;                   // open for reading
    let i_queries_group = file.group("itest")?;     // in distribution queries

    let i_queries_knns = i_queries_group.dataset("knns").unwrap();  // 11000 X 1000
    // let i_queries_dists = i_queries_group.dataset("dists").unwrap();
    let knns_rows = i_queries_knns.shape()[0];
    let knns_cols = i_queries_knns.shape()[1];

    if num_records_required > knns_rows {
        error!("Too many records requested" )
    }
    let num_records = if num_records_required == 0 { knns_rows } else { num_records_required.min(knns_rows) } as usize;
    let num_cols = if num_knns_required == 0 { knns_cols } else { num_knns_required.min(knns_cols) } as usize;

    let knns: Array2<usize> = i_queries_knns.read_slice(s![0..num_records, ..]).unwrap(); // get num_records records

    Ok(filter_knns(knns,num_records,num_cols).unwrap())
}

// Filter out all entries that are greater than or equal to num_records (rows run from 0..num_records)
// then only keep the first num_cols of data
fn filter_knns(knns: Array2<usize>, num_records: usize, num_cols: usize) -> anyhow::Result<Array2<usize>> {
    // Vector to collect cleaned rows
    let mut filtered_rows: Vec<Array1<usize>> = Vec::new();

    for row in knns.rows() {
        // Filter those values < num_records
        let filtered: Vec<usize> = row.iter()
            .filter_map(|x| if x < &num_records { Some(x) } else { None } )
            .take(num_cols)                     // Take only first num_cols after filtering
            .cloned()
            .collect();

        if filtered.len() < num_cols {
            // Not enough entries — return error or handle as needed
            println!( "Not enough enties {} {}", filtered.len(), num_cols );
            return Err(anyhow::anyhow!("Row does not contain enough valid entries"));
        }

        filtered_rows.push(Array1::from(filtered));
    }

    // Stack all rows vertically to form an Array2
    let required_knns = ndarray::stack(Axis(0), &filtered_rows.iter().map(|x| x.view()).collect::<Vec<_>>())?;

    Ok(required_knns)
}